import os
import numpy as np
from sklearn.model_selection import train_test_split
import tifffile as tiff
from scipy.interpolate import interpn
import time
from disp_cell3D import plot4d # Custom function to plot 4D data

# Set path to the data folder
data_folder = "../cellpol/orig"
file_names = sorted(os.listdir(data_folder), key=lambda x: x.split('_')[0])

# Load the data and masks from the files
data = []
masks = []

# Process each filename and load the 4D image from the file
for file_name in file_names:
    if file_name.endswith('_mask.tif'):
        mask_path = os.path.join(data_folder, file_name)
        mask = tiff.imread(mask_path)
        masks.append(mask)
    elif file_name.endswith('.tif'):
        image_path = os.path.join(data_folder, file_name)
        image = tiff.imread(image_path)
        data.append(image)
# Ensure that data and masks have the same length
assert len(data) == len(masks)

# Find the maximum and average shape of the data and masks
max_shape = [0,0,0,0]
min_shape = [1000,1000,1000,1000]
temp_total = [0,0,0,0]
for i in range(len(data)):
     temp = data[i].shape
     for j in range(len(temp)):
        temp_total[j] += temp[j]
        if temp[j] > max_shape[j]:
            max_shape[j] = temp[j]
        elif temp[j] < min_shape[j]:
            min_shape[j] = temp[j]
avg_shape = [int(np.ceil(x / len(data))) for x in temp_total]
print('Max shape:', max_shape)
print('Average shape:', avg_shape)
print('Min shape:', min_shape)

#Preprocess the data
# Normalize the data.
for i in range(len(data)):
    data[i] = data[i]/data[i].max() # Normalize the data
    masks[i] = np.bool_(masks[i]) # Convert to boolean to save memory

# Resize the data and masks
new_shape = (32,32,32)
data_ds = np.zeros((len(data),20,32,32,32))
masks_ds = np.zeros((len(data),20,32,32,32))

# Time the loop
start_time = time.time()
for i in range(len(data)):
    # Select only the middle 20 time steps
    for t in range((data[i].shape[0]+1)//2-9, data[i].shape[0]//2+10):
        # Downsample the data using trilinear interpolation
        data_3d = data[i][t, :, :, :]
        data_3d_range = (np.arange(data_3d.shape[0]), np.arange(data_3d.shape[1]), np.arange(data_3d.shape[2]))
        ds_coords = np.mgrid[0:data_3d.shape[0]:new_shape[0]*1j, 0:data_3d.shape[1]:new_shape[1]*1j, 0:data_3d.shape[2]:new_shape[2]*1j].T
        data_ds_temp = np.float32(interpn(data_3d_range, data_3d, ds_coords, method='linear', bounds_error=False, fill_value=0))
        # Downsample the mask using nearest neighbor interpolation
        mask_3d = masks[i][t, :, :, :]
        mask_3d_range = data_3d_range
        mask_ds_temp = np.bool_(interpn(mask_3d_range, mask_3d, ds_coords, method='nearest', bounds_error=False, fill_value=0))
        
        data_ds[i,t-(data[i].shape[0]+1)//2-9,:,:,:] = data_ds_temp
        masks_ds[i,t-(data[i].shape[0]+1)//2-9,:,:,:] = mask_ds_temp
end_time = time.time()
print('Time taken:', end_time - start_time)
data_ds = np.reshape(data_ds, (data_ds.shape[0]*data_ds.shape[1], data_ds.shape[2], data_ds.shape[3], data_ds.shape[4]))
masks_ds = np.reshape(masks_ds, (masks_ds.shape[0]*masks_ds.shape[1], masks_ds.shape[2], masks_ds.shape[3], masks_ds.shape[4]))
X_train, X_test, y_train, y_test = train_test_split(data_ds, masks_ds, test_size=0.2, random_state=42)

if not os.path.exists('CNN_data'):
    os.mkdir('CNN_data')
np.save(os.path.join('CNN_data','X_train.npy'), X_train)
np.save(os.path.join('CNN_data','X_test.npy'), X_test)
np.save(os.path.join('CNN_data','y_train.npy'), y_train)
np.save(os.path.join('CNN_data','y_test.npy'), y_test)