import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels):
        super(UNet, self).__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
        )
        in_channels *= 2

        for i in range(num_levels - 1):
            self.downsamples.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding="same"),
                    nn.ReLU(),
                ),
            )
            in_channels *= 2
        

        self.middle = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        for i in range(num_levels - 1, 0, -1):
            self.upsamples.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels * 2, in_channels // 2, kernel_size=2, stride=2),
                    nn.ReLU(),
                )
            )
            in_channels //= 2

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding="same"),
        )
    def forward(self, x):
        b, c, h, w = x.size()
        skips = []

        x = self.in_conv(x)
        skips.append(x)

        # Downsample and cross-attention
        for i in range(self.num_levels - 1):
            x = self.downsamples[i](x)
            skips.append(x)
        
        x = self.middle(x)

        # Upsample and concatenate with skips and cross-attention
        for i, skip in enumerate(reversed(skips[1:])):
            x = torch.cat([x, skip], dim=1)
            x = self.upsamples[i](x)

        # Output convolution
        x = torch.cat([x, skips[0]], dim=1)
        x = self.out_conv(x)
        return x