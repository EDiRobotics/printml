import os
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from printml.Datasets.DeformDataset import DeformDataset
from printml.Datasets.DataPrefetcher import DataPrefetcher
from printml.Wrappers.DeformationPredictor import DeformationPredictor
from printml.Train import train

if __name__ == '__main__':
    
    # Saving path
    save_path = Path('./Save/')
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    file_name = 'fake_deform_data.hdf5'
    dataset_path = Path('/root/autodl-tmp/printml/printml') / file_name
    bs_per_gpu = 128
    workers_per_gpu = 4
    cache_ratio = 2

    # Network
    num_levels = 8
    num_cross_attn_levels = 3
    head_dim = 64
    n_heads = 8

    # Training
    num_training_epochs = 50
    save_interval = 5
    load_epoch_id = 0
    gradient_accumulation_steps = 1
    lr_max = 1e-4
    do_watch_parameters = False
    do_profile = False

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
        num_cross_attn_levels=num_cross_attn_levels,
        head_dim=head_dim,
        n_heads=n_heads,
        device=device,
    )
    predictor.load_pretrained(acc, save_path, load_epoch_id)
    predictor.load_wandb(acc, save_path, do_watch_parameters, save_interval)
    optimizer = torch.optim.Adam(
        predictor.net.parameters(),
        lr=lr_max,
        fused=True,
    )
    predictor.net, optimizer, loader = acc.prepare(
        predictor.net, 
        optimizer, 
        loader, 
        device_placement=[True, True, False],
    )
    prefetcher = DataPrefetcher(loader, device)

    train(
        acc, 
        prefetcher, 
        predictor,
        optimizer, 
        num_training_epochs,
        save_path,
        load_epoch_id,
        save_interval,
        do_profile,
    )