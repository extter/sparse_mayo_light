import torch
from torch import nn



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        out = out + residual
        out = self.relu_out(out)

        return out

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, block_cls=DoubleConv):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = block_cls(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, block_cls=DoubleConv):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = block_cls(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[-2:], mode='bilinear', align_corners=False
            )

        x = torch.cat([skip, x], dim=1)
        return self.block(x)
