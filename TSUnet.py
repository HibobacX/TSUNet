import torch
import torch.nn as nn
from mam import EWFM
import torch.nn.functional as F

##########################################################################
# Basic modules
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.cwa = EWFM(n_feat=n_feat, kernel_size=3, reduction=16, bias=bias, act=act)  # EWFM模块

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.cwa(res)
        res += x
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Compute inter-stage features
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x1 = x1 + x
        return x1, img


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=5):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()  # []
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(
                UNetConvBlock(in_size=n_feat + scale_unetfeats * i, out_size=n_feat + scale_unetfeats * (i + 1),
                              downsample=True, relu_slope=0.2, use_HIN=True))
        self.body.append(UNetConvBlock(in_size=n_feat + scale_unetfeats * (depth - 1),
                                       out_size=n_feat + scale_unetfeats * (depth - 1), downsample=False,
                                       relu_slope=0.2, use_HIN=True))

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x, encoder_outs[i], decoder_outs[-i - 1])
                    res.append(x_up)
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x)
                    res.append(x_up)
                else:
                    x = down(x)
        return res, x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN
        self.residual = nn.Conv2d(in_size, out_size, 1, 1, 0)
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)
        self.conv_block = EWFM(n_feat=in_size, kernel_size=3, reduction=16, bias=False, act=nn.PReLU())

    def forward(self, x):
        out = self.conv_block(x)
        out = self.conv_1(out)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.residual(x)
        out += self.identity(x)

        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(out_size * 2, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=5):
        super(Decoder, self).__init__()

        self.body = nn.ModuleList()
        self.skip_conv = nn.ModuleList()  # []
        for i in range(depth - 1):
            self.body.append(UNetUpBlock(in_size=n_feat + scale_unetfeats * (depth - i - 1),
                                         out_size=n_feat + scale_unetfeats * (depth - i - 2), relu_slope=0.2))
            self.skip_conv.append(
                nn.Conv2d(n_feat + scale_unetfeats * (depth - i - 1), n_feat + scale_unetfeats * (depth - i - 2), 3, 1,
                          1))

    def forward(self, x, bridges):

        res = []
        for i, up in enumerate(self.body):
            x = up(x, self.skip_conv[i](bridges[-i - 1]))
            res.append(x)

        return res




class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, act):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.act = act
        self.alpha = nn.Parameter(torch.Tensor([0.7]))

    def forward(self, x):
        return self.act(self.conv(x))



##########################################################################
## DGUNet_plus
class TSUNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3,
                 reduction=8,bias=False, depth=5):
        super(TSUNet, self).__init__()

        act = nn.PReLU()
        self.depth = depth
        self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                           DilatedConvBlock(n_feat, n_feat, kernel_size=3, dilation=2, padding=2,
                                                             act=nn.ReLU(inplace=True))
                                           )
        self.shallow_feat2 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                           DilatedConvBlock(n_feat, n_feat, kernel_size=3, dilation=2, padding=2,
                                                            act=nn.ReLU(inplace=True))
                                           )
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)

        self.phi_0 = ResBlock(default_conv, 3, 3)
        self.phit_0 = ResBlock(default_conv, 3, 3)
        self.phi_1 = ResBlock(default_conv, 3, 3)
        self.phit_1 = ResBlock(default_conv, 3, 3)
        self.r0 = nn.Parameter(torch.Tensor([0.5]))
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        # self.merge67 = mergeblock(n_feat, 3, True)
        self.concat67 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        # self.tail = conv(n_feat, 3, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, 3, kernel_size, bias=bias)

    def forward(self, img):
        res = []
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_1 = self.phi_0(img) - img
        x1_img = img - self.r0 * self.phit_0(phixsy_1)
        ## PMM
        x1 = self.shallow_feat1(x1_img)
        feat1, feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)
        res.append(stage1_img)
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        phixsy_2 = self.phi_1(stage1_img) - img
        x2_img = stage1_img - self.r1 * self.phit_1(phixsy_2)
        # PMM
        x2 = self.shallow_feat2(x2_img)
        # x2_cat = self.merge67(x2, x2_samfeats)
        # stage2_img = self.tail(x2_cat) + img
        x2_cat = self.concat67(torch.cat([x2, x2_samfeats], 1))
        stage2_img = self.tail(x2_cat)+ img
        return [stage2_img, stage1_img]
