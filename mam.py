import torch
import torch.nn as nn
import torch.nn.functional as F
from WT import DWT, IWT


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False, n_feat=3):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)



class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )
        # 添加一个门控卷积层，输出大小为1，即一个标量
        self.gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),  # 使用1x1卷积作为门控
            nn.Sigmoid()
        )

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]

        # 计算空间注意力权重
        attention_weights = self.conv_du(channel_pool)
        # 计算门控权重
        gate_weights = self.gate(channel_pool)

        # 应用门控调制
        modulated_attention = attention_weights * gate_weights
        alpha = nn.Parameter(torch.tensor(0.5))
        res = x * modulated_attention + alpha * x  # 残差连接
        return res


##########################################################################
##---------- Enhanced Wavelet- Domain Feature Fusion Module (EWFM) Blocks ----------
class EWFM(nn.Module):
    def __init__(self, n_feat=64, kernel_size=3, reduction=16, bias=False, act=nn.PReLU()):
        super(EWFM, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        modules_body = [
            DepthwiseSeparableConv(n_feat * 2, n_feat, kernel_size, padding=(kernel_size - 1) // 2, bias=bias)]
        for _ in range(3):  # Increasing depth here
            modules_body.append(act)
            modules_body.append(
                DepthwiseSeparableConv(n_feat, n_feat, kernel_size, padding=(kernel_size - 1) // 2, bias=bias))

        modules_body.append(
            DepthwiseSeparableConv(n_feat, n_feat * 2, kernel_size, padding=(kernel_size - 1) // 2, bias=bias))

        self.body = nn.Sequential(*modules_body)
        self.WSA = SALayer()
        self.CurCA = CurveCALayerrh(n_feat * 2)
        self.conv1x1 = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, bias=bias)  # 256 to 128
        self.conv3x3 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        residual = x
        _, _, h, w = x.size()
        pad_h = (h + 1) // 2 * 2 - h
        pad_w = (w + 1) // 2 * 2 - w
        x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain (Dual attention)
        x_dwt = self.dwt(wavelet_path_in)

        res = self.body(x_dwt)
        branch_sa = self.WSA(res)
        branch_curveca_2 = self.CurCA(res)
        res = torch.cat([branch_sa, branch_curveca_2], dim=1)
        res = self.conv1x1(res) + x_dwt

        wavelet_path = self.iwt(res)
        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.activate(self.conv3x3(out))
        out = out[..., :h, :w]
        out += self.conv1x1_final(residual)

        return out


class LightweightMultiScaleFeatureFusion(nn.Module):
    def __init__(self, channels, reduction=4):
        super(LightweightMultiScaleFeatureFusion, self).__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, groups=channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=3, padding=1, groups=channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=5, padding=2, groups=channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        feat_1x1 = self.conv_1x1(x)
        feat_3x3 = self.conv_3x3(x)
        feat_5x5 = self.conv_5x5(x)
        output = feat_1x1 + feat_3x3 + feat_5x5
        return output


class DepthwiseSeparableConvone(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConvone, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CurveCALayerrh(nn.Module):
    def __init__(self, channel):
        super(CurveCALayerrh, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.n_curve = 3
        self.relu = nn.ReLU(inplace=False)

        # 使用多尺度特征融合模块
        self.multi_scale_feature_fusion = LightweightMultiScaleFeatureFusion(channel)


        self.predict_a = nn.Sequential(
            DepthwiseSeparableConvone(channel, channel, 5, padding=2), nn.ReLU(inplace=True),
            DepthwiseSeparableConvone(channel, channel, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 应用多尺度特征融合
        x = self.multi_scale_feature_fusion(x)

        a = self.predict_a(x)
        x = self.relu(x) - self.relu(x - 1)
        for i in range(self.n_curve):
            x = x + a[:, i:i + 1] * x * (1 - x)
        return x