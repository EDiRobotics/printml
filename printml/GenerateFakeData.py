"""
Generate an HDF dataset with fake data.
A sample includes
    - temperature, shape: (HEIGHT, WIDTH), generated using random numer
    - altitude, shape: (HEIGHT, WIDTH), generated using random numer
    - thickness, shape: (HEIGHT, WIDTH), generated using random numer
    - trajectory, shape: (timestep, 2), randomly generated 2D trajectories
    - deformation, shape: (HEIGHT, WIDTH). Path integrals for temperature, HEIGHT, and thickness along trajectory.
"""

import numpy as np
import h5py
from tqdm import tqdm
from printml.DeformDataset import HEIGHT, WIDTH

if __name__ == '__main__':
    # Define the dimensions and number of samples
    timestep = 50
    sample_num = 10000  # Number of samples
    neighborhood_size = 3  # Size of the neighborhood to average around the path

    # Create an HDF5 file
    with h5py.File('fake_deform_data.hdf5', 'w') as hdf_file:
        # Create datasets with an extra dimension for the number of samples
        temp_dataset = hdf_file.create_dataset('temperature', (sample_num, HEIGHT, WIDTH)) #, compression='lzf')
        alt_dataset = hdf_file.create_dataset('altitude', (sample_num, HEIGHT, WIDTH)) #, compression='lzf')
        thick_dataset = hdf_file.create_dataset('thickness', (sample_num, HEIGHT, WIDTH)) #, compression='lzf')
        traj_dataset = hdf_file.create_dataset('trajectory', (sample_num, timestep, 2)) #, compression='lzf')
        def_dataset = hdf_file.create_dataset('deformation', (sample_num, HEIGHT, WIDTH)) #, compression='lzf')

        # Generate and store data for each sample
        for i in tqdm(range(sample_num), desc="creating samples"):
            # Generate random data for the current sample
            temperature = np.random.rand(HEIGHT, WIDTH)
            altitude = np.random.rand(HEIGHT, WIDTH)
            thickness = np.random.rand(HEIGHT, WIDTH)

            # Generate a trajectory within the range of (HEIGHT, WIDTH) using float numbers
            trajectory = np.stack([
                np.random.uniform(0, HEIGHT, size=timestep),
                np.random.uniform(0, WIDTH, size=timestep)
            ], axis=-1)

            # Initialize deformation
            deformation = np.zeros((HEIGHT, WIDTH))

            # Calculate deformation using data around the path
            for t in range(timestep):
                y, x = trajectory[t].astype(int)  # Convert to int for indexing

                # Define the neighborhood around the current point on the trajectory
                y_min = max(0, y - neighborhood_size // 2)
                y_max = min(HEIGHT, y + neighborhood_size // 2 + 1)
                x_min = max(0, x - neighborhood_size // 2)
                x_max = min(WIDTH, x + neighborhood_size // 2 + 1)

                # Calculate the average value in the neighborhood
                temp_avg = np.mean(temperature[y_min:y_max, x_min:x_max])
                alt_avg = np.mean(altitude[y_min:y_max, x_min:x_max])
                thick_avg = np.mean(thickness[y_min:y_max, x_min:x_max])

                # Update the deformation map
                deformation[y_min:y_max, x_min:x_max] += temp_avg + alt_avg + thick_avg

            # Store the data in the respective datasets
            temp_dataset[i] = temperature
            alt_dataset[i] = altitude
            thick_dataset[i] = thickness
            traj_dataset[i] = trajectory
            def_dataset[i] = deformation

    print("HDF5 dataset with multiple fake samples and neighborhood-averaged deformation has been created.")