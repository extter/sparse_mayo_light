
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(
        self,
        data_range=1.0,
        window_size=11,
        sigma=1.5,
        channels=1,
        size_average=True,
        k1=0.01,
        k2=0.03,
        eps=1e-8,
    ):
        super().__init__()
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.channels = int(channels)
        self.size_average = size_average
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.eps = float(eps)

        window = self._create_gaussian_window(self.window_size, self.sigma, self.channels)
        self.register_buffer("window", window)

    def _create_gaussian_window(self, window_size, sigma, channels):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        kernel_2d = torch.outer(g, g)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = kernel_2d.view(1, 1, window_size, window_size)
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
        return kernel_2d

    def _get_window(self, x):
        c = x.shape[1]
        if c == self.window.shape[0]:
            return self.window.to(device=x.device, dtype=x.dtype)

        window = self._create_gaussian_window(self.window_size, self.sigma, c)
        return window.to(device=x.device, dtype=x.dtype)

    def forward(self, x, y):
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError(f"SSIM expects 4D tensors [B, C, H, W], got {x.shape} and {y.shape}")
        if x.shape != y.shape:
            raise ValueError(f"SSIM expects same shape, got {x.shape} and {y.shape}")

        x = x.float()
        y = y.float()

        window = self._get_window(x)
        c = x.shape[1]
        padding = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=padding, groups=c)
        mu_y = F.conv2d(y, window, padding=padding, groups=c)

        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=c) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=c) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=padding, groups=c) - mu_xy

        sigma_x2 = torch.clamp(sigma_x2, min=0.0)
        sigma_y2 = torch.clamp(sigma_y2, min=0.0)

        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2

        num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)

        ssim_map = num / (den + self.eps)

        if self.size_average:
            return ssim_map.mean()
        return ssim_map.mean(dim=(1, 2, 3))

class MixedLoss(nn.Module):
    def __init__(self, w_ssim=0.8, w_l1=0.15, w_mse=0.05):
        super().__init__()
        self.w_ssim = w_ssim
        self.w_l1 = w_l1
        self.w_mse = w_mse
        self.ssim = SSIM(data_range=1.0, window_size=11, sigma=1.5, channels=1, size_average=True)

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        pred = pred.float()
        target = target.float()

        ssim_loss = 1.0 - self.ssim(pred, target)
        l1_loss = F.l1_loss(pred, target)
        mse_loss = F.mse_loss(pred, target)

        return self.w_ssim * ssim_loss + self.w_l1 * l1_loss + self.w_mse * mse_loss