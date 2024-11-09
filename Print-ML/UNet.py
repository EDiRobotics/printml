import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels, out_channels, num_levels):
        super(UNet, self).__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.cross_attentions = nn.ModuleList()
        self.condition_projectors = nn.ModuleList()

        for i in range(num_levels):
            self.downsamples.append(nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2))
            self.cross_attentions.append(Attention(query_dim=in_channels * 2, context_dim=2))
            in_channels *= 2

        self.upsamples.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
        in_channels //= 2
        for i in range(num_levels - 2, -1, -1):
            self.upsamples.append(nn.ConvTranspose2d(in_channels * 2, in_channels // 2, kernel_size=2, stride=2))
            in_channels //= 2

        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, cond):
        b, c, h, w = x.size()
        skips = []

        # Downsample and cross-attention
        for i in range(self.num_levels):
            x = self.downsamples[i](x)
            x_flat = x.flatten(2).permute(0, 2, 1)
            x_flat = self.cross_attentions[i](x_flat, cond)
            x = x_flat.permute(0, 2, 1).reshape(x.shape)
            skips.append(x)

        # Upsample and concatenate with skips
        x = self.upsamples[0](x)
        for i, skip in enumerate(reversed(skips[:-1])):
            x = torch.cat([x, skip], dim=1)
            x = self.upsamples[i+1](x)

        # Output convolution
        x = self.out_conv(x)
        return x

if __name__ == '__main__':
    # Usage example
    batch_size = 5
    height = 256
    width = 256
    in_channels = 5
    out_channels = 1
    seq_length = 10  # Length of the trajectory sequence

    # Generate random input data
    img = torch.randn(batch_size, in_channels, height, width)
    trajectory = torch.randn(batch_size, seq_length, 2)

    # Initialize the UNet model
    unet = UNet(in_channels=in_channels, out_channels=out_channels, num_levels=4)

    # Forward pass
    deformation_output = unet(img, trajectory)

    # Print the shape of the output to verify
    print("Output shape:", deformation_output.shape)