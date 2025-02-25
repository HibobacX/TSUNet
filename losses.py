import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        kernel = torch.Tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        self.kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def forward(self, x, y):
        x_edge = F.conv2d(x, self.kernel, padding=1, groups=3)
        y_edge = F.conv2d(y, self.kernel, padding=1, groups=3)
        loss = self.loss(x_edge, y_edge)
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


from torchvision.models import vgg16


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg16(pretrained=True).features
        self.vgg = nn.Sequential(*list(vgg.children())[:16])  # 例如使用前16层
        self.vgg.eval()  # 设置为评估模式
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = nn.functional.l1_loss(x_vgg, y_vgg)  # 使用L1损失计算特征的差异
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, lambda_vgg=1.0, lambda_l1=1.0):
        super(CombinedLoss, self).__init__()
        self.lambda_vgg = lambda_vgg
        self.lambda_l1 = lambda_l1
        self.vgg_loss = VGGLoss()
        self.l1_loss = CharbonnierLoss()

    def forward(self, generated, target):
        loss_vgg = self.vgg_loss(generated, target)
        loss_l1 = self.l1_loss(generated, target)
        total_loss = self.lambda_vgg * loss_vgg + self.lambda_l1 * loss_l1
        return total_loss

