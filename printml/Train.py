"""
A train loop
"""

import os
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from printml.DeformDataset import DeformDataset
from printml.UNet import DeformationPredictor

if __name__ == '__main__':
    
    # Saving path
    save_path = Path('./Save/')
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    file_name = 'fake_deform_data.hdf5'
    dataset_path = Path('/root/autodl-tmp/printml/printml') / file_name
    bs_per_gpu = 64
    workers_per_gpu = 4
    cache_ratio = 2

    # Network
    num_levels = 4
    head_dim = 64
    n_heads = 8

    # Training
    num_training_epochs = 50
    save_interval = 50
    load_epoch_id = 0
    gradient_accumulation_steps = 1
    lr_max = 1e-4
    warmup_steps = 5
    weight_decay = 1e-4
    max_grad_norm = 10
    print_interval = 10
    do_watch_parameters = False

    # Preparation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        kwargs_handlers=[kwargs],
    )
    device = acc.device
    dataset = DeformDataset(str(dataset_path))
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=bs_per_gpu,
        num_workers=workers_per_gpu,
        prefetch_factor=cache_ratio,
    )
    predictor = DeformationPredictor(
        num_levels=num_levels,
        head_dim=head_dim,
        n_heads=n_heads,
        device=device,
    )
    predictor.load_pretrained(acc, save_path, load_epoch_id)
    predictor.load_wandb(acc, save_path, do_watch_parameters, save_interval)
    optimizer = torch.optim.Adam(
        predictor.net.parameters(),
        lr=lr_max,
    )
    predictor.net, optimizer, loader = acc.prepare(
        predictor.net, 
        optimizer, 
        loader, 
        device_placement=[True, True, True],
    )

    # Training loop
    for epoch in range(num_training_epochs):
        total_loss = 0
        for batch in loader:
            with acc.accumulate(predictor.net):
                predictor.net.train()
                loss = predictor.compute_loss(batch)
                optimizer.zero_grad()
                acc.backward(loss)
                optimizer.step()
                total_loss += loss.item()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Average Loss: {average_loss}")

        # Log the average loss to wandb
        acc.log({"Average Loss": average_loss})

        # Save a checkpoint every "save_interval" epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f"checkpoint_{epoch+1}.pth")