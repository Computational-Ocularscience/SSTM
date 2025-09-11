import torch
import torch.nn as nn
import torch.nn.functional as F
from gma import Aggregate

##3D GRU UpdateBlock
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        #try temporal here
        self.convz1 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,5), padding=(0,0,2))
        self.convr1 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,5), padding=(0,0,2))
        self.convq1 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,5), padding=(0,0,2))

        self.convz2 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,5,1), padding=(0,2,0))
        self.convr2 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,5,1), padding=(0,2,0))
        self.convq2 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,5,1), padding=(0,2,0))

        self.convz3 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,1), padding=0)
        self.convr3 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,1), padding=0)
        self.convq3 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,1), padding=0)




    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        # temporal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz3(hx))
        r = torch.sigmoid(self.convr3(hx))
        q = torch.tanh(self.convq3(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

# class SmallMotionEncoder(nn.Module):
#     def __init__(self, args):
#         super(SmallMotionEncoder, self).__init__()
#         cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
#         self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
#         self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
#         self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
#         self.conv = nn.Conv2d(128, 80, 3, padding=1)
#
#     def forward(self, flow, corr):
#         cor = F.relu(self.convc1(corr))
#         flo = F.relu(self.convf1(flow))
#         flo = F.relu(self.convf2(flo))
#         cor_flo = torch.cat([cor, flo], dim=1)
#         out = F.relu(self.conv(cor_flo))
#         return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        #self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf1 = nn.Conv2d(3, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr, err):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        flo = torch.cat([flow, err], dim=1)

        flo = F.relu(self.convf1(flo))

        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)

        out = F.relu(self.conv(cor_flo))

        return torch.cat([out, flow], dim=1)

    # class SmallUpdateBlock(nn.Module):
    #     def __init__(self, args, hidden_dim=96):
    #         super(SmallUpdateBlock, self).__init__()
    #         self.encoder = SmallMotionEncoder(args)
    #         self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
    #         self.flow_head = FlowHead(hidden_dim, hidden_dim=128)
    #
    #     def forward(self, net, inp, corr, flow):
    #         motion_features = self.encoder(flow, corr)
    #         inp = torch.cat([inp, motion_features], dim=1)
    #         net = self.gru(net, inp)
    #         delta_flow = self.flow_head(net)
    #
    #         return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+128)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)

    def forward(self, net, inp1, inp2, corr1, corr2, flow1, flow2, attention1,attention2, er1, er2, upsample=True):

        motion_features1 = self.encoder(flow1, corr1, er1)
        motion_features2 = self.encoder(flow2, corr2, er2)

        motion_features_global1 = self.aggregator(attention1, motion_features1)
        motion_features_global2 = self.aggregator(attention2, motion_features2)

        inp1 = torch.cat([inp1, motion_features1, motion_features_global1], dim=1)
        inp2 = torch.cat([inp2, motion_features2, motion_features_global2], dim=1)


        b, d, h, w = inp1.shape
        inp1 = inp1.view(b, d, 1, h, w)
        inp2 = inp2.view(b, d, 1, h, w)
        inp = torch.cat([inp1, inp2], dim=2)

        net = self.gru(net, inp)
        net1, net2 = torch.split(net, [1, 1], dim=2)
        b, d, _, h, w = net1.shape

        net1 = net1.view(b, d, h, w)
        net2 = net2.view(b, d, h, w)

        delta_flow1 = self.flow_head(net1)
        delta_flow2 = self.flow_head(net2)

        # scale mask to balence gradients
        mask1 = .25 * self.mask(net1)
        mask2 = .25 * self.mask(net2)
        return net, mask1, mask2, delta_flow1, delta_flow2








