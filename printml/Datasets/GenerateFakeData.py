"""
Generate an HDF dataset with fake data.
A sample includes
    - temperature, shape: (HEIGHT, WIDTH), generated using random numer
    - altitude, shape: (HEIGHT, WIDTH), generated using random numer
    - thickness, shape: (HEIGHT, WIDTH), generated using random numer
    - trajectory, shape: (timestep, 2), randomly generated 2D trajectories
    - deformation, shape: (HEIGHT, WIDTH). Path integrals for temperature, HEIGHT, and thickness along trajectory.
"""

import random
import numpy as np
import h5py
from tqdm import tqdm
from printml.Datasets.DeformDataset import HEIGHT, WIDTH

if __name__ == '__main__':
    # Define the dimensions and number of samples
    timestep = 10000
    sample_num = 10000  # Number of samples
    neighborhood_size = 5  # Size of the neighborhood to average around the path

    # Create an HDF5 file
    with h5py.File('fake_deform_data.hdf5', 'w') as hdf_file:
        # Create datasets with an extra dimension for the number of samples
        temp_dataset = hdf_file.create_dataset('temperature', (sample_num, HEIGHT, WIDTH)) #, compression='lzf')
        alt_dataset = hdf_file.create_dataset('altitude', (sample_num, HEIGHT, WIDTH)) #, compression='lzf')
        thick_dataset = hdf_file.create_dataset('thickness', (sample_num, HEIGHT, WIDTH)) #, compression='lzf')
        traj_dataset = hdf_file.create_dataset('trajectory', (sample_num, timestep, 2), dtype="i") #, compression='lzf')
        energy_dataset = hdf_file.create_dataset('energy', (sample_num, timestep)) #, compression='lzf')
        def_dataset = hdf_file.create_dataset('deformation', (sample_num, HEIGHT, WIDTH)) #, compression='lzf')

        # Generate and store data for each sample
        for i in tqdm(range(sample_num), desc="creating samples"):
            # Generate random data for the current sample
            temperature = np.random.rand(HEIGHT, WIDTH).astype(np.float32)
            altitude = np.random.rand(HEIGHT, WIDTH).astype(np.float32)
            thickness = np.random.rand(HEIGHT, WIDTH).astype(np.float32)
            energy = np.random.rand(timestep).astype(np.float32)

            # 自动选择参数
            start_point = np.array([np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)])  # 随机选择起始点
            move_along_row = np.random.choice([True, False])  # 随机选择是沿行移动还是沿列移动
            if move_along_row:
                row_width = np.random.randint(1, WIDTH - start_point[0] + 1)  # 随机选择行宽
                row_spacing = np.random.randint(1, HEIGHT - start_point[1] + 1)  # 随机选择行间距
            else:
                row_width = np.random.randint(1, HEIGHT - start_point[1] + 1)  # 随机选择列宽
                row_spacing = np.random.randint(1, WIDTH - start_point[0] + 1)  # 随机选择列间距

            # 初始化轨迹数组
            trajectory = np.zeros((timestep, 2), dtype=np.int32)

            # 设置当前点为起始点
            current_point = start_point.copy()
            trajectory[0] = current_point

            # 生成轨迹
            for t in range(1, timestep):
                if move_along_row:
                    # 沿行移动
                    if t % (2 * row_width) < row_width:
                        # 从左到右移动
                        current_point[0] += 1
                    else:
                        # 从右到左移动
                        current_point[0] -= 1
                    # 每行走完一行后，随机选择上移或下移
                    if t % (2 * row_width) == 0:
                        move_direction = np.random.choice([-1, 1])  # 随机选择上移（-1）或下移（1）
                        current_point[1] += move_direction * row_spacing
                else:
                    # 沿列移动
                    if t % (2 * row_width) < row_width:
                        # 从上到下移动
                        current_point[1] += 1
                    else:
                        # 从下到上移动
                        current_point[1] -= 1
                    # 每行走完一列后，随机选择左移或右移
                    if t % (2 * row_width) == 0:
                        move_direction = np.random.choice([-1, 1])  # 随机选择左移（-1）或右移（1）
                        current_point[0] += move_direction * row_spacing

                # 确保点在定义的范围内
                current_point = np.clip(current_point, [0, 0], [WIDTH - 1, HEIGHT - 1])
                if current_point[0] == HEIGHT - 1 or current_point[0] == 0 or current_point[1] == WIDTH - 1 or current_point[1] == 0:
                    energy[t] = 0

                # 更新轨迹
                trajectory[t] = current_point

            # Initialize deformation
            deformation = np.zeros((HEIGHT, WIDTH))

            # Calculate deformation using data around the path
            for t in range(timestep):
                y, x = trajectory[t] 

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
                deformation[y_min:y_max, x_min:x_max] += (temp_avg + alt_avg + thick_avg) * energy[t]

            # Store the data in the respective datasets
            temp_dataset[i] = temperature
            alt_dataset[i] = altitude
            thick_dataset[i] = thickness
            traj_dataset[i] = trajectory
            energy_dataset[i] = energy
            def_dataset[i] = deformation

            """
            # Pick a sample to visualize using matplotlib
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(temperature, cmap='hot')
            plt.title('Temperature')

            plt.subplot(1, 4, 2)
            plt.imshow(altitude, cmap='hot')
            plt.title('Altitude')

            plt.subplot(1, 4, 3)
            plt.imshow(thickness, cmap='hot')
            plt.title('Thickness')

            plt.subplot(1, 4, 4)
            plt.imshow(deformation, cmap='hot')
            plt.title('Deformation')

            # 使用plt.savefig()来保存整个图表
            plt.savefig(f'sample_{i}.png')
            import pdb; pdb.set_trace()
            """

    print("HDF5 dataset with multiple fake samples and neighborhood-averaged deformation has been created.")