"""
A train loop
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

def train(
    acc, 
    prefetcher,
    predictor,
    optimizer, 
    num_training_epochs,
    save_path,
    load_epoch_id,
    save_interval,
    do_profile,
):
    if do_profile:
        prof = profile(
            schedule = torch.profiler.schedule(
                wait=20,
                warmup=3,
                active=4,
                repeat=1,
            ),
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=tensorboard_trace_handler(save_path/'prof'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
        prof.start()

    batch_idx = 0
    for epoch in range(num_training_epochs):
        total_loss = 0
        batch, _ = prefetcher.next()
        while batch is not None:
            with acc.accumulate(predictor.net):
                predictor.net.train()
                loss = predictor.compute_loss(batch)
                optimizer.zero_grad()
                acc.backward(loss)
                optimizer.step()
                total_loss += loss.detach()
                batch_idx += 1
            batch, _ = prefetcher.next()
            
            if do_profile:
                prof.step()
                if batch_idx == 28:
                    prof.stop()
                    acc.print("Profiling log saved in ", str(save_path/'prof'))
                    acc.print("Visualize the profiling log by tensorboard with torch_tb_profiler plugin, see https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html")

        # Calculate average loss for the epoch
        average_loss = total_loss / len(prefetcher.loader)
        print(f"Epoch {epoch+1}, Average Loss: {average_loss}")

        # Log the average loss to wandb
        acc.log({"Average Loss": average_loss})

        if epoch % save_interval == 0:
            predictor.save_pretrained(acc, save_path, epoch+load_epoch_id)