# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:33:00 2024

@author: jacke
"""
import nibabel as nib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ants
import time
import os
import sys
def get_file_path(filename):
    if hasattr(sys, '_MEIPASS'):
        # If running in a PyInstaller bundle, look in the temporary directory
        return os.path.join(sys._MEIPASS, filename)
    else:
        # Otherwise, look in the current working directory
        return os.path.join(os.getcwd(), filename)

def nii_gz_to_numpy(file_path):
    """
    Loads a .nii.gz file and converts it to a NumPy array.

    Parameters:
    - file_path (str): The path to the .nii.gz file.

    Returns:
    - data (np.ndarray): The image data as a NumPy array.
    - affine (np.ndarray): The affine transformation matrix.
    - header (nibabel.Nifti1Header): The header of the NIfTI file.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        # Load the NIfTI file
        nii_img = nib.load(file_path)
    except Exception as e:
        raise IOError(f"An error occurred while loading the NIfTI file: {e}")

    # Get the image data as a NumPy array
    data = nii_img.get_fdata()

    return data


# Define the list of mask names in the order they appear in the NIfTI file
mask_names = [
    "R_Cingulate_Ant",
    "L_Cingulate_Ant",
    "R_Cingulate_Post",
    "L_Cingulate_Post",
    "R_Insula",
    "L_Insula",
    "R_Brainstem",
    "L_Brainstem",
    "R_Thalamus",
    "L_Thalamus",
    "R_Caudate",
    "L_Caudate",
    "R_Putamen",
    "L_Putamen",
    "R_Pallidum",
    "L_Pallidum",
    "R_Substantia_nigra",
    "L_Substantia_nigra",
    "R_Frontal_Lat",
    "L_Frontal_Lat",
    "R_Orbital",
    "L_Orbital",
    "R_Frontal_Med_Sup",
    "L_Frontal_Med_Sup",
    "R_Precentral",
    "L_Precentral",
    "R_Parietal_Inf",
    "L_Parietal_Inf",
    "R_Postcentral",
    "L_Postcentral",
    "R_Precuneus",
    "L_Precuneus",
    "R_Parietal_Sup",
    "L_Parietal_Sup",
    "R_Temporal_Mesial",
    "L_Temporal_Mesial",
    "R_Temporal_Basal",
    "L_Temporal_Basal",
    "R_Temporal_Lat_Ant",
    "L_Temporal_Lat_Ant",
    "R_Occipital_Med",
    "L_Occipital_Med",
    "R_Occipital_Lat",
    "L_Occipital_Lat",
    "R_Cerebellum",
    "L_Cerebellum",
    "R_Vermis",
    "L_Vermis"
]

# Path to the NIfTI file containing all masks
# file_path = 'brain_masks.nii'

import os
import sys
import nibabel as nib

# Function to determine the correct file path
def get_file_path(filename):
    if hasattr(sys, '_MEIPASS'):
        # If running in a PyInstaller bundle, look in the temporary directory
        return os.path.join(sys._MEIPASS, filename)
    else:
        # Otherwise, look in the current working directory
        return os.path.join(os.getcwd(), filename)

# Get the file path for the .nii file
file_path = get_file_path('brain_masks.nii')

# Load the NIfTI file using nibabel
nifti_img = nib.load(file_path)
data = nifti_img.get_fdata()

def regions_z_score(k_1, frame, wat):
    # Load the transformed K_1 array
    # k_1 = np.load('transformed_K_1.npy')
    
    # Verify that the number of masks matches the number of mask names
    num_masks = data.shape[-1]
    if num_masks != len(mask_names):
        raise ValueError(f"Number of masks in NIfTI file ({num_masks}) does not match number of mask names provided ({len(mask_names)}).")
    
    # Initialize a dictionary to store mean values for each region
    mean_values = {}
    
    
    # Iterate over each mask and compute the mean value
    for i, mask_name in enumerate(mask_names):
        mask = data[..., i]
        
        # Ensure that the mask is binary
        if not np.array_equal(mask, mask.astype(bool)):
            # print(f"Warning: Mask '{mask_name}' is not binary. Proceeding with computation.")
        
        # Apply the transformation k_1 to the mask
        transformed_mask = mask * k_1
        
        # Calculate the sum of transformed mask values
        sum_transformed = np.sum(transformed_mask)
        
        # Calculate the number of voxels in the mask
        sum_mask = np.sum(mask)
        
        # Avoid division by zero
        if sum_mask == 0:
            mean_val = np.nan  # or any other placeholder value
            # print(f"Warning: Mask '{mask_name}' contains no voxels. Mean value set to NaN.")
        else:

            transformed_mask[transformed_mask<0.1]=np.nan
            mean_val = np.nanmean(transformed_mask)
        
        # Store the mean value in the dictionary
        mean_values[mask_name] = mean_val
        
        
        # # print the mean value
        # print(f"Mean value for {mask_name}: {mean_val:.10f}")
    #%%
    # Let's read the data directly from the text file named "Mean perfusion with slope and intersect"
    if frame=="real" and wat== 'wat':
        file_path = get_file_path('48_region_mean_std_k1.txt')
    elif  frame=="ref" and wat=='wat':
        file_path = get_file_path('48_region_mean_std_k1_ref.txt')
    elif frame=='real' and wat=='wat1':
        file_path = get_file_path('48_region_mean_std_k1_1.txt')
    elif frame=='ref' and wat=='wat1':
        file_path = get_file_path('48_region_mean_std_k1_1_ref.txt')
    elif frame=='real' and wat=='wat2':
        file_path = get_file_path('48_region_mean_std_k1_2.txt')
    elif frame=='ref' and wat=='wat2':
        file_path = get_file_path('48_region_mean_std_k1_2_ref.txt')
    elif frame=='real' and wat=='wat1_2':
        file_path = get_file_path('48_region_mean_std_k1_3.txt')
    elif frame=='ref' and wat=='wat1_2':
        file_path = get_file_path('48_region_mean_std_k1_3_ref.txt')
    
    # Reading the text file and parsing it
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data_lines = [line.strip() for line in lines if not line.startswith('%') and line.strip()]

    # # Skip the header row
    # data_lines = data_lines[1:]
    
    # Initialize lists for the columns
    regions, loaded_means, loaded_stds = [], [], []
    
    for line in data_lines:
        parts = line.split(',')
        regions.append(parts[0].strip())
        loaded_means.append(float(parts[1].strip()))
        loaded_stds.append(float(parts[2].strip()))
    
    # Create a DataFrame from the parsed data
    df_from_file = pd.DataFrame({
        'Region': regions,
        'Mean': loaded_means,
        'Std': loaded_stds,
        # 'Slope': slopes,
        # 'Intersect': intersects
    })
    
    means=np.zeros(np.shape(data[:,:,:,0]))
    z_scores=[]
    for i, mask_name in enumerate(mask_names):
        # # print(mean_values[mask_name])
        means=means+((mean_values[mask_name]-loaded_means[i])/loaded_stds[i])*data[..., i]
        z_score = (mean_values[mask_name] - loaded_means[i]) / loaded_stds[i]
        z_scores.append(z_score)
        # means=means+(i*data[..., i]-loaded_means[i])/loaded_stds[i]
        # # print(mean_values[mask_name]*data[..., i])
    #%%
    # for i in range(128):
    #     plt.imshow(np.rot90(means[:,:,i]), vmin=0, vmax=1)
    #     plt.axis('off')
    #     plt.colorbar()
    #     plt.show()
    means[means == 0] = np.nan
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 1), (1 ,1 ,1), (1, 0, 0), (0.5, 0, 0)]  # R, G, B
    n_bins = 10  # Number of bins in the colormap
    
    # Create the colormap
    cmap_name = 'custom_blue_white_red'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    def view_angels(image_array):
        x_dim, y_dim, z_dim = image_array.shape
        first_values_xz_pos_y = np.full((x_dim, z_dim), np.nan)
        first_values_xz_neg_y = np.full((x_dim, z_dim), np.nan)
        
        # Arrays for the x-axis values
        first_values_yz_pos_x = np.full((y_dim, z_dim), np.nan)
        first_values_yz_neg_x = np.full((y_dim, z_dim), np.nan)
        
        # Find first non-NaN value along positive x-axis
        for y in range(y_dim):
            for z in range(z_dim):
                first_valid_idx = np.where(~np.isnan(image_array[:, y, z]))[0]
                if first_valid_idx.size > 0:
                    first_values_yz_pos_x[y, z] = image_array[first_valid_idx[0], y, z]
        
        # Find first non-NaN value along negative x-axis
        for y in range(y_dim):
            for z in range(z_dim):
                first_valid_idx = np.where(~np.isnan(image_array[::-1, y, z]))[0]
                if first_valid_idx.size > 0:
                    first_values_yz_neg_x[y, z] = image_array[-(first_valid_idx[0] + 1), y, z]
        
        # Find first non-NaN value along positive y-axis
        for x in range(x_dim):
            for z in range(z_dim):
                first_valid_idx = np.where(~np.isnan(image_array[x, :, z]))[0]
                if first_valid_idx.size > 0:
                    first_values_xz_pos_y[x, z] = image_array[x, first_valid_idx[0], z]
                    
        # Find first non-NaN value along negative y-axis
        for x in range(x_dim):
            for z in range(z_dim):
                first_valid_idx = np.where(~np.isnan(image_array[x, ::-1, z]))[0]
                if first_valid_idx.size > 0:
                    first_values_xz_neg_y[x, z] = image_array[x, -(first_valid_idx[0] + 1), z]
       
        return first_values_yz_pos_x, first_values_yz_neg_x, first_values_xz_pos_y, first_values_xz_neg_y
    
    means_right=means.copy()
    means_left=means.copy()
    for z in range(128):
        for x in range(63,128):
            for y in range(153):
                means_right[x,y,z]=np.nan
    for z in range(128):
        for x in range(65):
            for y in range(153):
                means_left[x,y,z]=np.nan
    first_values_yz_pos_x0, first_values_yz_neg_x0, first_values_xz_pos_y0, first_values_xz_neg_y0= view_angels(means_right)
    first_values_yz_pos_x1, first_values_yz_neg_x1, first_values_xz_pos_y1, first_values_xz_neg_y1= view_angels(means_left)
    first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2= view_angels(means)
    first_values_xz_neg_y2=np.flip(first_values_xz_neg_y2, axis=0)
    
    
    def plot_first_values(vmin=None, vmax=None):
        fig, axes = plt.subplots(1, 6, figsize=(20, 6))
        
        # Plot for the first non-NaN value along positive x-axis
        im1 = axes[0].imshow(np.rot90(np.rot90(np.rot90(first_values_yz_neg_x0))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title('Dex inside')
        axes[0].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative x-axis
        im2 = axes[1].imshow(np.flip(np.rot90(np.rot90(np.rot90(first_values_yz_pos_x1))), axis=1), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title('Sin inside)')
        axes[1].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along positive y-axis
        im3 = axes[2].imshow(np.flip(np.rot90(np.rot90(np.rot90(first_values_yz_pos_x2))), axis=1), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[2].set_title('Dex')
        axes[2].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative y-axis
        im4 = axes[3].imshow(np.rot90(np.rot90(np.rot90(first_values_yz_neg_x2))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[3].set_title('Sin')
        axes[3].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along positive y-axis
        im5 = axes[4].imshow(np.rot90(np.rot90(np.rot90(first_values_xz_pos_y2))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[4].set_title('Back')
        axes[4].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative y-axis
        im6 = axes[5].imshow(np.rot90(np.rot90(np.rot90(first_values_xz_neg_y2))), interpolation='nearest', cmap=custom_cmap, vmin=vmin, vmax=vmax)
        axes[5].set_title('Front')
        axes[5].axis('off')  # Remove axis
    
        # Add one shared colorbar for all subplots
        fig.subplots_adjust(right=0.85)  # Adjust to leave space for colorbar
        cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im1, cax=cbar_ax, label='ml/cm3/min')
    
        plt.tight_layout()
        plt.show()
    
    # Example usage:
    plot_first_values(vmin=-5, vmax=5)
    return first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2, z_scores, means, mean_values

# first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2 = regions_z_score()
# k_1=np.load('transformed_k_1_1.npy')
# frame='real'
# wat='wat1'
# regions_z_score(k_1, frame, wat)