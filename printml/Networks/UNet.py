import torch
import torch.nn as nn
import torch.nn.functional as F

TRAJ_DIM = 2
IN_CHANNELS = 5
OUT_CHANNELS = 1

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, head_dim=64, n_heads=8, drop_p=0.1, causal=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.causal = causal
        self.q_net = nn.Linear(query_dim, head_dim * n_heads)
        self.k_net = nn.Linear(context_dim, head_dim * n_heads)
        self.v_net = nn.Linear(context_dim, head_dim * n_heads)
        self.proj_net = nn.Linear(head_dim * n_heads, query_dim)
        self.drop_p = drop_p

    def forward(self, x, context):
        B, T, _ = x.shape # batch size, seq length, ff_dim
        E, D = self.n_heads, self.head_dim

        # Divide the tensors for multi head dot product
        q = self.q_net(x).view(B, T, E, D).transpose(1, 2) # b t (e d) -> b e t d
        k = self.k_net(context).view(B, -1, E, D).transpose(1, 2) # b t (e d) -> b e t d
        v = self.v_net(context).view(B, -1, E, D).transpose(1, 2) # b t (e d) -> b e t d

        inner = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_p, is_causal=self.causal)
        inner = inner.transpose(1, 2).contiguous().view(B, T, E * D) # b e t d -> b t (e d)
        return self.proj_net(inner)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels, num_cross_attn_levels, head_dim, n_heads):
        super(UNet, self).__init__()
        self.num_levels = num_levels
        self.num_cross_attn_levels = num_cross_attn_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.down_cross_attentions = nn.ModuleList()
        self.up_cross_attentions = nn.ModuleList()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
        )
        in_channels *= 2

        for i in range(num_levels - 1):
            self.downsamples.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding="same"),
                    nn.BatchNorm2d(in_channels * 2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, padding="same"),
                    nn.BatchNorm2d(in_channels * 2),
                    nn.ReLU(),
                ),
            )
            if i > num_levels - num_cross_attn_levels - 1:
                self.down_cross_attentions.append(
                    Attention(
                        query_dim=in_channels * 2, 
                        context_dim=TRAJ_DIM,
                        head_dim=head_dim,
                        n_heads=n_heads,
                    ),
                )
            in_channels *= 2
        

        self.middle = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        for i in range(num_levels - 1, 0, -1):
            self.upsamples.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels * 2, in_channels // 2, kernel_size=2, stride=2),
                    nn.BatchNorm2d(in_channels // 2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding="same"),
                    nn.BatchNorm2d(in_channels // 2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding="same"),
                    nn.BatchNorm2d(in_channels // 2),
                    nn.ReLU(),
                )
            )
            if i > num_levels - num_cross_attn_levels - 1:
                self.up_cross_attentions.append(
                    Attention(
                        query_dim=in_channels // 2, 
                        context_dim=TRAJ_DIM,
                        head_dim=head_dim,
                        n_heads=n_heads,
                    ),
                )
            in_channels //= 2

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x, cond):
        b, c, h, w = x.size()
        skips = []

        x = self.in_conv(x)
        skips.append(x)

        # Downsample and cross-attention
        for i in range(self.num_levels - self.num_cross_attn_levels):
            x = self.downsamples[i](x)
            skips.append(x)
        
        for i in range(self.num_levels - self.num_cross_attn_levels, self.num_levels - 1):
            x = self.downsamples[i](x)
            x_flat = x.flatten(2).permute(0, 2, 1)
            x_flat = self.down_cross_attentions[i - (self.num_levels - self.num_cross_attn_levels)](x_flat, cond)
            x = x_flat.permute(0, 2, 1).reshape(x.shape)
            skips.append(x)
        
        x = self.middle(x)

        # Upsample and concatenate with skips and cross-attention
        for i, skip in enumerate(reversed(skips[-self.num_cross_attn_levels:])):
            x = torch.cat([x, skip], dim=1)
            x = self.upsamples[i](x)
            x_flat = x.flatten(2).permute(0, 2, 1)
            x_flat = self.up_cross_attentions[i](x_flat, cond)
            x = x_flat.permute(0, 2, 1).reshape(x.shape)

        for i, skip in enumerate(reversed(skips[1:-self.num_cross_attn_levels])):
            x = torch.cat([x, skip], dim=1)
            x = self.upsamples[i+self.num_cross_attn_levels](x)

        # Output convolution
        x = torch.cat([x, skips[0]], dim=1)
        x = self.out_conv(x)
        return x