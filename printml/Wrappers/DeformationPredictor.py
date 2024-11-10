import os
import json
import torch
import torch.nn.functional as F
import wandb
from printml.Datasets.DeformDataset import HEIGHT, WIDTH
from printml.Networks.UNet import UNet, IN_CHANNELS, OUT_CHANNELS

class DeformationPredictor():

    def __init__(self, num_levels, head_dim, n_heads, device):
        self.net = UNet(IN_CHANNELS, OUT_CHANNELS, num_levels, head_dim, n_heads).to(device)
        self.x_coords = torch.linspace(0, 1, steps=WIDTH).unsqueeze(0).repeat(HEIGHT, 1).to(device)
        self.y_coords = torch.linspace(0, 1, steps=HEIGHT).unsqueeze(1).repeat(1, WIDTH).to(device)
        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.net.parameters()))
        )
    
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