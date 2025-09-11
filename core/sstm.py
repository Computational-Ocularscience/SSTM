import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from update import BasicUpdateBlock
from extractor import BasicEncoder
from corr import CorrBlock, AlternateCorrBlock
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from utils.utils import bilinear_sampler, coords_grid, upflow8
from gma import Attention, Aggregate
from spt import ErrorBlock

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class SSTM(nn.Module):
    def __init__(self, args):
        super(SSTM, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)

        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)

        self.errorfunc = ErrorBlock()


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, image3, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between triplets of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image3 = 2 * (image3 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        image3 = image3.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2, fmap3 = self.fnet([image1, image2, image3])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        fmap3 = fmap3.float()


        if self.args.alternate_corr:
            corr_fn1 = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            corr_fn2 = AlternateCorrBlock(fmap2, fmap3, radius=self.args.corr_radius)
        else:
            corr_fn1 = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            corr_fn2 = CorrBlock(fmap2, fmap3, radius=self.args.corr_radius)


        # run the context network
        with autocast(enabled=self.args.mixed_precision):

            cmap1, cmap2, _= self.cnet([image1, image2, image3])

            net1, inp1 = torch.split(cmap1, [hdim, cdim], dim=1)
            net2, inp2 = torch.split(cmap2, [hdim, cdim], dim=1)

            net1 = torch.tanh(net1)
            inp1 = torch.relu(inp1)
            net2 = torch.tanh(net2)
            inp2 = torch.relu(inp2)

            b, d, h, w = net1.shape
            net1 = net1.view(b,d,1,h,w)
            net2 = net2.view(b,d,1,h,w)

            net = torch.cat([net1, net2], dim=2)


            attention1 = self.att(inp1)
            attention2 = self.att(inp2)

        coords00, coords01 = self.initialize_flow(image1)
        coords10, coords11 = self.initialize_flow(image2)
        if flow_init is not None:
            coords01 = coords01 + flow_init
            coords11 = coords11

        flow_predictions1 = []
        flow_predictions2 = []

        flow_low1 = []
        flow_low2 = []


        inplist1 = []
        inplist2 = []

        for itr in range(iters):

            coords01 = coords01.detach()
            coords11 = coords11.detach()

            corr1 = corr_fn1(coords01)
            corr2 = corr_fn2(coords11)

            flow1 = coords01 - coords00
            flow2 = coords11 - coords10

            er1, er2 = self.errorfunc([flow1, flow2],[fmap1, fmap2, fmap3])


            if itr+1%4==0:
                inp1 = inp1 + inplist1[(itr + 1)//4 - 1]
                inp2 = inp2 + inplist2[(itr + 1)//4 - 1]

            with autocast(enabled=self.args.mixed_precision):
                net, up_mask1, up_mask2, delta_flow1, delta_flow2 = self.update_block(net, inp1, inp2, corr1, corr2, flow1, flow2, attention1,attention2, er1, er2)

            if itr%4==0:
                inplist1.append(inp1)
                inplist2.append(inp2)

            # F(t+1) = F(t) + \Delta(t)

            coords01 = coords01 + delta_flow1
            coords11 = coords11 + delta_flow2

            # upsample predictions
            if up_mask1 is None or up_mask2 is None:
                #flow_up = upflow8(coords1 - coords0)
                flow_up1 = upflow8(coords01 - coords00)
                flow_up2 = upflow8(coords11 - coords10)
            else:
                #flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                flow_up1 = self.upsample_flow(coords01 - coords00, up_mask1)
                flow_up2 = self.upsample_flow(coords11 - coords10, up_mask2)


            flow_predictions1.append(flow_up1)
            flow_predictions2.append(flow_up2)

            flow_low1.append(coords01 - coords00)
            flow_low2.append(coords11 - coords10)


        if test_mode:
            return flow_low1, flow_low2, flow_up1, flow_up2
            # return flow_low1, flow_low2, flow_up1, flow_up2, flow_predictions1, flow_predictions2
            #return coords01 - coords00, coords11 - coords10, flow_up1, flow_up2, flow_predictions1, flow_predictions2
            #return coords00, coords10, flow_up1, flow_up2
            
        #return flow_predictions
        return [flow_predictions1, flow_predictions2]