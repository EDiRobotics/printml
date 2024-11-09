import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from printml.DeformDataset import HEIGHT, WIDTH

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
    def __init__(self, in_channels, out_channels, num_levels, head_dim, n_heads):
        super(UNet, self).__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.cross_attentions = nn.ModuleList()
        self.condition_projectors = nn.ModuleList()

        for i in range(num_levels):
            self.downsamples.append(
                nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2),
            )
            self.cross_attentions.append(
                Attention(
                    query_dim=in_channels * 2, 
                    context_dim=TRAJ_DIM,
                    head_dim=head_dim,
                    n_heads=n_heads,
                ),
            )
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

class DeformationPredictor():

    def __init__(self, num_levels, head_dim, n_heads, device):
        self.net = UNet(IN_CHANNELS, OUT_CHANNELS, num_levels, head_dim, n_heads)
        self.x_coords = torch.linspace(0, 1, steps=WIDTH).unsqueeze(0).repeat(HEIGHT, 1).to(device)
        self.y_coords = torch.linspace(0, 1, steps=HEIGHT).unsqueeze(1).repeat(1, WIDTH).to(device)
    
    def save_pretrained(self, acc, path, epoch_id):
        acc.wait_for_everyone()
        acc.save(acc.unwrap_model(self.net).state_dict(), path / f'net_{epoch_id}.pth')

    def load_pretrained(self, acc, path, load_epoch_id):
        if os.path.isfile(path / f'net_{load_epoch_id}.pth'):
            ckpt = torch.load(path / f'net_{load_epoch_id}.pth', map_location='cpu', weights_only=True)
            missing_keys, unexpected_keys = self.net.load_state_dict(ckpt, strict=False)
            acc.print('load ', path / f'net_{load_epoch_id}.pth', '\nmissing ', missing_keys, '\nunexpected ', unexpected_keys)
        else: 
            acc.print(path / f'net_{load_epoch_id}.pth', 'does not exist. Initialize new checkpoint')

    def load_wandb(self, acc, path, do_watch_parameters, save_interval):
        if os.path.isfile(path / "wandb_id.json"):
            run_id = json.load(open(path / "wandb_id.json", "r"))
            acc.init_trackers(
                project_name="printml", 
                init_kwargs={"wandb": {"id": run_id, "resume": "allow"}}
            )
            if acc.is_main_process:
                if do_watch_parameters:
                    wandb.watch(self.net, log="all", log_freq=save_interval)
        else: 
            acc.init_trackers(project_name="printml")
            if acc.is_main_process:
                tracker = acc.get_tracker("wandb")
                json.dump(tracker.run.id, open(path / "wandb_id.json", "w"))
                if do_watch_parameters:
                    wandb.watch(self.net, log="all", log_freq=save_interval)
    
    def compute_loss(self, batch):
        B = batch["temperature"].shape[0]
        img = torch.stack(
            (
                batch["temperature"],
                batch["altitude"],
                batch["thickness"],
                self.x_coords[None].repeat(B, 1, 1),
                self.y_coords[None].repeat(B, 1, 1),
            ), 
            dim=1,
        )
        deformation = self.net(img, batch["trajectory"])
        loss = F.l1_loss(deformation, batch["deformation"][:, None])
        return loss