import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class SpatialFeatureEncoder(nn.Module):
    def __init__(self, in_planes, planes):
        super(SpatialFeatureEncoder, self).__init__()

        self.conv11 = nn.Conv2d(in_planes, planes, (7, 7), stride=2, padding=3)
        self.conv12 = nn.Conv2d(planes, planes, (3, 3), stride=1, padding=1)
        self.conv13 = nn.Conv2d(planes, planes, (3, 3), stride=1, padding=1)
        self.conv14 = nn.Conv2d(planes, planes, (3, 3), stride=1, padding=1)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)
        self.norm3 = nn.BatchNorm2d(planes)
        self.norm4 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.relu(self.norm1(self.conv11(x)))
        x = self.relu(self.norm2(self.conv12(x)))
        x = self.relu(self.norm3(self.conv13(x)))
        x = self.relu(self.norm4(self.conv14(x)))

        return x

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, in_planes, planes, norm_fn, spt_type, stride):
        super(SpatioTemporalEncoder, self).__init__()

        self.spt_type = spt_type

        self.convr1 = nn.Conv3d(in_planes, planes, kernel_size=1)
        self.convs1 = nn.Conv3d(planes, planes, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.convt1 = nn.Conv3d(planes, planes, (3,1,1), stride=1, padding=(1,0,0))
        self.convr2 = nn.Conv3d(planes, planes, kernel_size=1, stride = (1,stride,stride))

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.downsample = nn.Sequential(nn.Conv3d(planes, planes, kernel_size=1, stride=(1,2,2)), nn.BatchNorm3d(planes))

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=planes // 8, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=planes // 8, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=planes // 8, num_channels=planes)
            self.norm4 = nn.GroupNorm(num_groups=planes // 8, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm3d(planes)
            self.norm2 = nn.BatchNorm3d(planes)
            self.norm3 = nn.BatchNorm3d(planes)
            self.norm4 = nn.BatchNorm3d(planes)

    def spt_1(self, x):
        x = self.relu(self.norm1(self.convr1(x)))
        s1 = self.relu(self.norm2(self.convs1(x)))
        t1 = self.relu(self.norm3(self.convt1(s1)))
        out = self.relu(self.norm4(self.convr2(t1)))
        if self.stride==2:
            x = self.downsample(x)
        return self.relu(out+x)

    def spt_2(self, x):
        x = self.relu(self.norm1(self.convr1(x)))
        s1 = self.relu(self.norm2(self.convs1(x)))
        t1 = self.relu(self.norm3(self.convt1(x)))
        out = self.relu(self.norm4(self.convr2(s1 + t1)))
        if self.stride==2:
            x = self.downsample(x)
        return self.relu(out+x)

    def spt_3(self, x):
        x = self.relu(self.norm1(self.convr1(x)))
        s1 = self.relu(self.norm2(self.convs1(x)))
        t1 = self.relu(self.norm3(self.convt1(s1)))
        out = self.relu(self.norm4(self.convr2(t1 + s1)))
        if self.stride==2:
            x = self.downsample(x)
        return self.relu(out+x)

    def spt_4(self, x):
        x = self.relu(self.norm1(self.convr1(x)))
        t1 = self.relu(self.norm2(self.convt1(x)))
        s1 = self.relu(self.norm3(self.convs1(t1)))
        out = self.relu(self.norm4(self.convr2(t1 + s1)))
        if self.stride==2:
            x = self.downsample(x)
        return self.relu(x+out)



    def forward(self, x):

        if self.spt_type == 'SPT1':
            st = self.spt_1(x)
        elif self.spt_type == 'SPT2':
            st = self.spt_2(x)
        elif self.spt_type == 'SPT3':
            st = self.spt_3(x)
        elif self.spt_type == 'SPT4':
            st = self.spt_4(x)

        return st

class MotionFeatureBlock(nn.Module):
    def __init__(self, in_planes, planes, down_sample=None):
        super(MotionFeatureBlock, self).__init__()

        self.downsample = down_sample

        self.conv11 = nn.Conv3d(3, 64, (3, 7, 7), stride=(1,2,2), padding=(1,3,3))
        self.conv12 = nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1)
        self.conv13 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(128, 256, (1,1,1), stride=(2,1,1), padding=0)

        self.relu = nn.ReLU(inplace=True)

        self.SpatioTemp11 = SpatioTemporalEncoder(in_planes=64, planes=64, norm_fn='batch', spt_type='SPT1', stride=1)
        self.SpatioTemp21 = SpatioTemporalEncoder(in_planes=64, planes=64, norm_fn='batch', spt_type='SPT2', stride=2)
        self.SpatioTemp41 = SpatioTemporalEncoder(in_planes=64, planes=96, norm_fn='batch', spt_type='SPT3', stride=1)
        self.SpatioTemp22 = SpatioTemporalEncoder(in_planes=96, planes=96, norm_fn='batch', spt_type='SPT1', stride=1)
        self.SpatioTemp32 = SpatioTemporalEncoder(in_planes=96, planes=128, norm_fn='batch', spt_type='SPT4', stride=2)

        self.norm1 = nn.BatchNorm3d(64)
        self.norm2 = nn.BatchNorm3d(64)
        self.norm3 = nn.BatchNorm3d(64)
        self.norm4 = nn.BatchNorm3d(256)

        self.downsample = nn.Sequential(nn.Conv3d(64, 128, kernel_size=1, stride=(1, 2, 2)), nn.BatchNorm3d(128),
                                        nn.Conv3d(128, 128, kernel_size=1, stride=(1, 2, 2)), nn.BatchNorm3d(128))

    def forward(self, img1, img2, img3):
        x = [img1, img2, img3]

        b, c, h, w = x[0].shape

        img1 = x[0].view(b, c, 1, h, w)
        img2 = x[1].view(b, c, 1, h, w)
        img3 = x[2].view(b, c, 1, h, w)

        x = torch.cat([img1, img2, img3], dim=2)

        x = self.relu(self.norm1(self.conv11(x)))
        x = self.relu(self.norm2(self.conv12(x)))
        x = self.relu(self.norm3(self.conv13(x)))

        res = x

        y = self.SpatioTemp11(x)
        y = self.SpatioTemp21(y)
        y = self.SpatioTemp41(y)

        y = self.SpatioTemp22(y)
        y = self.SpatioTemp32(y)


        res = self.downsample(res)
        out = self.relu(self.norm4(self.conv2(y+res)))

        return out


class Warpping(nn.Module):
    def __init__(self):
        super(Warpping, self).__init__()

    def forward(self, flo, fmap2):

        B, C, H, W = fmap2.shape

        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.cuda()

        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(fmap2, vgrid, align_corners=True)

        mask = torch.autograd.Variable(torch.ones(fmap2.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        fmap1_estimate = output * mask


        return fmap1_estimate


class ErrorBlock(nn.Module):
    def __init__(self, in_planes=2, planes=1):
        super(ErrorBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1), nn.BatchNorm2d(planes))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1), nn.BatchNorm2d(planes))

    def warp(self, flo, fmap2):

        B, C, H, W = fmap2.shape
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.cuda()

        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(fmap2, vgrid, align_corners=True)

        mask = torch.autograd.Variable(torch.ones(fmap2.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        fmap1_estimate = output * mask

        return fmap1_estimate

    def forward(self, flow, fmap):

        fmap1est = self.warp(flow[0], fmap[1])
        fmap2est = self.warp(flow[1], fmap[2])
        fmap11est = self.warp(flow[0], fmap2est)

        bright_err1 = torch.norm(fmap[0]-fmap1est, p=2, dim=1, keepdim=True)
        bright_err2 = torch.norm(fmap[1]-fmap2est, p=2, dim=1, keepdim=True)
        bright_err3 = torch.norm(fmap[0]-fmap11est, p=2, dim=1, keepdim=True)

        bright_err13 = torch.cat([bright_err1, bright_err3], dim=1)
        bright_err23 = torch.cat([bright_err2, bright_err3], dim=1)


        err1 = self.conv1(bright_err13)
        err2 = self.conv2(bright_err23)

        return err1, err2







