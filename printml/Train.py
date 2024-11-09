"""
A train loop
"""

import torch
from printml.DeformDataset import DeformDataset
from printml.UNet import DeformationPredictor

dataset = DeformDataset('fake_deform_data.hdf5')
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=4,
)
predictor = DeformationPredictor(
    num_levels=4, 
    head_dim=64, 
    n_heads=8,
)
optimizer = torch.optim.Adam(
    predictor.net.parameters(),
    lr=1e-4,
)

for epoch in range(100):
    for batch in dataloader:
        loss = predictor.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)