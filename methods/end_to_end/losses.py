
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import SSIM

class MixedLoss(nn.Module):
    def __init__(self, w_ssim = 0.8, w_l1 = 0.15, w_MSE = 0.05):
        super().__init__()
        self.w_ssim = w_ssim
        self.w_l1 = w_l1
        self.w_MSE = w_MSE

    def forward(self, pred, target):
        ssim_loss = 1 -SSIM(data_range=1, size_average=True)(pred, target)
        l1_loss = F.l1_loss(pred, target)
        MSE_loss = F.mse_loss(pred, target)

        return self.w_ssim * ssim_loss + self.w_l1 * l1_loss + self.w_MSE * MSE_loss










    
