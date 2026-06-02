from torch import nn

from .blocks import DoubleConv, DownBlock, UpBlock, ResidualDoubleConv


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, 2 * base_ch)
        self.enc3 = DownBlock(2 * base_ch, 4 * base_ch)
        self.bottleneck = DownBlock(4 * base_ch, 8 * base_ch)
        self.dec3 = UpBlock(8 * base_ch, 4 * base_ch, 4 * base_ch)
        self.dec2 = UpBlock(4 * base_ch, 2 * base_ch, 2 * base_ch)
        self.dec1 = UpBlock(2 * base_ch, base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        h = self.bottleneck(s3)
        h = self.dec3(h, s3)
        h = self.dec2(h, s2)
        h = self.dec1(h, s1)
        return self.out_conv(h)


class ResidualUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()

        self.enc1 = ResidualDoubleConv(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, 2 * base_ch, block_cls=ResidualDoubleConv)
        self.enc3 = DownBlock(2 * base_ch, 4 * base_ch, block_cls=ResidualDoubleConv)

        self.bottleneck = DownBlock(4 * base_ch, 8 * base_ch, block_cls=ResidualDoubleConv)

        self.dec3 = UpBlock(8 * base_ch, 4 * base_ch, 4 * base_ch, block_cls=ResidualDoubleConv)
        self.dec2 = UpBlock(4 * base_ch, 2 * base_ch, 2 * base_ch, block_cls=ResidualDoubleConv)
        self.dec1 = UpBlock(2 * base_ch, base_ch, base_ch, block_cls=ResidualDoubleConv)

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        h = self.bottleneck(s3)

        h = self.dec3(h, s3)
        h = self.dec2(h, s2)
        h = self.dec1(h, s1)

        return self.out_conv(h)