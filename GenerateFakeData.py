"""
Generate an HDF dataset with fake data.
A sample includes
    - temperature, shape: (height, width), generated using random numer
    - altitude, shape: (height, width), generated using random numer
    - thickness, shape: (height, width), generated using random numer
    - trajectory, shape: (timestep, 2), randomly generated 2D trajectories
    - deformation, shape: (height, width). Path integrals for temperature, height, and thickness along trajectory.
"""

import numpy as np
import h5py
from tqdm import tqdm

if __name__ == '__main__':
    # Define the dimensions and number of samples
    height = 100
    width = 100
    timestep = 50
    sample_num = 10000  # Number of samples
    neighborhood_size = 3  # Size of the neighborhood to average around the path

    # Create an HDF5 file
    with h5py.File('fake_deform_data.hdf5', 'w') as hdf_file:
        # Create datasets with an extra dimension for the number of samples
        temp_dataset = hdf_file.create_dataset('temperature', (sample_num, height, width)) #, compression='lzf')
        alt_dataset = hdf_file.create_dataset('altitude', (sample_num, height, width)) #, compression='lzf')
        thick_dataset = hdf_file.create_dataset('thickness', (sample_num, height, width)) #, compression='lzf')
        traj_dataset = hdf_file.create_dataset('trajectory', (sample_num, timestep, 2)) #, compression='lzf')
        def_dataset = hdf_file.create_dataset('deformation', (sample_num, height, width)) #, compression='lzf')

        # Generate and store data for each sample
        for i in tqdm(range(sample_num), desc="creating samples"):
            # Generate random data for the current sample
            temperature = np.random.rand(height, width)
            altitude = np.random.rand(height, width)
            thickness = np.random.rand(height, width)

            # Generate a trajectory within the range of (height, width) using float numbers
            trajectory = np.stack([
                np.random.uniform(0, height, size=timestep),
                np.random.uniform(0, width, size=timestep)
            ], axis=-1)

            # Initialize deformation
            deformation = np.zeros((height, width))

            # Calculate deformation using data around the path
            for t in range(timestep):
                y, x = trajectory[t].astype(int)  # Convert to int for indexing

                # Define the neighborhood around the current point on the trajectory
                y_min = max(0, y - neighborhood_size // 2)
                y_max = min(height, y + neighborhood_size // 2 + 1)
                x_min = max(0, x - neighborhood_size // 2)
                x_max = min(width, x + neighborhood_size // 2 + 1)

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