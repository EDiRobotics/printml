import os
import json
import torch
import torch.nn.functional as F
import wandb
from printml.Datasets.DeformDataset import HEIGHT, WIDTH
from printml.Networks.UNet import UNet, IN_CHANNELS, OUT_CHANNELS

class DeformationPredictor():

    def __init__(self, num_levels, device):
        self.net = UNet(IN_CHANNELS, OUT_CHANNELS, num_levels).to(device)
        self.dummy_img = torch.zeros(1, HEIGHT, WIDTH, device=device)
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
        B, T, _ = batch["trajectory"].shape

        # Fill in traj img and energy img
        traj_img = self.dummy_img.clone().repeat(B, 1, 1)
        energy_img = self.dummy_img.clone().repeat(B, 1, 1)

        traj = batch["trajectory"].to(dtype=torch.int)
        batch_indices = torch.arange(B).repeat_interleave(T)
        y_indices = traj[:, :, 0].flatten()
        x_indices = traj[:, :, 1].flatten()

        traj_img[batch_indices, y_indices, x_indices] = 1
        energy_img[batch_indices, y_indices, x_indices] = batch["energy"].flatten()

        img = torch.stack(
            (
                batch["temperature"],
                batch["altitude"],
                batch["thickness"],
                traj_img,
                energy_img,
            ), 
            dim=1,
        )

        deformation = self.net(img)
        loss = F.l1_loss(deformation, batch["deformation"][:, None])
        return loss