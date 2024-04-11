import os
import numpy as np
from sklearn.model_selection import train_test_split
import tifffile as tiff
from scipy.interpolate import interpn
import time
import cv2

np.random.seed(42)

# Set path to the data folder
data_folder = os.path.join(os.getcwd(), "data")
file_names = sorted(os.listdir(data_folder), key=lambda x: x.split("_")[0])

# Load the data and masks from the files
data = []
masks = []

# Process each filename and load the 4D image from the file
for file_name in file_names:
    if file_name.endswith("_mask.tif"):
        mask_path = os.path.join(data_folder, file_name)
        mask = tiff.imread(mask_path)
        masks.append(mask)
    elif file_name.endswith(".tif"):
        image_path = os.path.join(data_folder, file_name)
        image = tiff.imread(image_path)
        data.append(image)
# Ensure that data and masks have the same length
assert len(data) == len(masks)

# Find the maximum and average shape of the data and masks
max_shape = [0, 0, 0, 0]
temp_total = [0, 0, 0, 0]
for i in range(len(data)):
    temp = data[i].shape
    for j in range(len(temp)):
        temp_total[j] += temp[j]
        if temp[j] > max_shape[j]:
            max_shape[j] = temp[j]

avg_shape = [int(np.ceil(x / len(data))) for x in temp_total]
print("Max shape:", max_shape)
print("Average shape:", avg_shape)

# Preprocess the data
# Normalize the data.
for i in range(len(data)):
    data[i] = (data[i] / data[i].max())  # Normalize the data
    masks[i] = np.bool_(masks[i])  # Convert to boolean to save memory

# Downsample the data and masks
# new_shape = (avg_shape[1], avg_shape[2], avg_shape[3])
new_shape = (32, 32, 32) # low resolution to speed up training
# samples = 2
# data = data[:samples]
# masks = masks[:samples]
data_ds = []
masks_ds = []
# Time the loop
start_time = time.time()
# Delete output directory if it exists
# if os.path.exists('output'):
#     os.system('rm -r output')
# if not os.path.exists('output'):
#     os.makedirs('output')
s = 0
for i in range(len(data)):
    n = 1
    exit_flag = 0
    random_indices = np.random.choice(data[i].shape[0], size=n, replace=False)
    while masks[i][random_indices].max() == False: # Ensure that the mask has at least one cell
        s += 1
        np.random.seed(s)
        random_indices = np.random.choice(data[i].shape[0], size=n, replace=False)
        exit_flag += 1
        if exit_flag > 100: # If the mask has no cell (or it takes too long to find), break the loop
            break
    for t in random_indices:
        # Downsample the data using trilinear interpolation
        data_3d = data[i][t, :, :, :]
        data_3d_range = (
            np.arange(data_3d.shape[0]),
            np.arange(data_3d.shape[1]),
            np.arange(data_3d.shape[2]),
        )
        ds_coords = np.mgrid[
            0 : data_3d.shape[0] : new_shape[0] * 1j,
            0 : data_3d.shape[1] : new_shape[1] * 1j,
            0 : data_3d.shape[2] : new_shape[2] * 1j,
        ].T
        data_ds_temp = np.float32(
            interpn(
                data_3d_range,
                data_3d,
                ds_coords,
                method="linear",
                bounds_error=False,
                fill_value=0,
            )
        )
        # Downsample the mask using nearest neighbor interpolation
        mask_3d = masks[i][t, :, :, :]
        mask_3d_range = data_3d_range
        mask_ds_temp = np.bool_(
            interpn(
                mask_3d_range,
                mask_3d,
                ds_coords,
                method="nearest",
                bounds_error=False,
                fill_value=0,
            )
        )

        # Create a directory for each time sample
        output_dir = "dataset/cellpol"
        data_dir = os.path.join(output_dir, "train" if i < 50 else "test", "data")
        mask_dir = os.path.join(output_dir, "train" if i < 50 else "test", "mask")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # Now for every slice, for every time sample, we have a 3D image and a 3D mask
        # We will take 2D slices from the 3D image and mask and save them as png files
        # We will create a separate folder for every time sample
        data_ds_temp = (65535 * data_ds_temp).astype(np.uint16)
        for z in range(data_ds_temp.shape[2]):
            data_ds = data_ds_temp[:, :, z]
            mask_ds = mask_ds_temp[:, :, z]
            # Save the 2D slice as a png file in the output directory
            cv2.imwrite(
                os.path.join(data_dir, f"im_{i}_time_{t}_slice_{z}.png"), data_ds
            )
            cv2.imwrite(
                os.path.join(mask_dir, f"mask_{i}_time_{t}_slice_{z}.png"),
                255 * mask_ds.astype(np.uint8),
            )

end_time = time.time()
print("Time taken:", end_time - start_time)
