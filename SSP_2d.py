# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:51:57 2024

@author: jacke
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable  # Corrected import
from numba import jit
import time
import os
import sys
import numpy as np

# Function to determine the correct file path
def get_file_path(filename):
    if hasattr(sys, '_MEIPASS'):
        # If running in a PyInstaller bundle, look in the temporary directory
        return os.path.join(sys._MEIPASS, filename)
    else:
        # Otherwise, look in the current working directory
        return os.path.join(os.getcwd(), filename)

cdict = {'red': ((0.0, 0.0, 0.0),
                  (0.1, 0.5, 0.5),
                  (0.2, 0.0, 0.0),
                  (0.4, 0.2, 0.2),
                  (0.6, 0.0, 0.0),
                  (0.8, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                  (0.1, 0.0, 0.0),
                  (0.2, 0.0, 0.0),
                  (0.4, 1.0, 1.0),
                  (0.6, 1.0, 1.0),
                  (0.8, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                  (0.1, 0.5, 0.5),
                  (0.2, 1.0, 1.0),
                  (0.4, 1.0, 1.0),
                  (0.6, 0.0, 0.0),
                  (0.8, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}

my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

plt.style.use('dark_background')

def SSP_2D(transformed_K_1): 
    K_1_reshape_list_np = np.array(transformed_K_1)
    K_1_reshape_list_np[np.isnan(K_1_reshape_list_np)] = 0
    # print("hejsa",np.max(K_1_reshape_list_np))
    # K_1_reshape_list_np=np.where(K_1_reshape_list_np==np.nan,0, K_1_reshape_list_np)
    ssp_t = time.time()
    volume_brain = np.array(transformed_K_1)

    @jit(nopython=True)
    def trilinear_interpolate(volume, x, y, z):
        x = min(max(x, 0), volume.shape[0] - 1)
        y = min(max(y, 0), volume.shape[1] - 1)
        z = min(max(z, 0), volume.shape[2] - 1)
        x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        x1 = min(x1, volume.shape[0] - 1)
        y1 = min(y1, volume.shape[1] - 1)
        z1 = min(z1, volume.shape[2] - 1)
        xd, yd, zd = x - x0, y - y0, z - z0
        c00 = volume[x0, y0, z0] * (1 - xd) + volume[x1, y0, z0] * xd
        c01 = volume[x0, y0, z1] * (1 - xd) + volume[x1, y0, z1] * xd
        c10 = volume[x0, y1, z0] * (1 - xd) + volume[x1, y1, z0] * xd
        c11 = volume[x0, y1, z1] * (1 - xd) + volume[x1, y1, z1] * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        c = c0 * (1 - zd) + c1 * zd
        return c

    def mean_along_vectors(volume, vectors_with_points, vector_length, num_steps):
        volume_2 = np.empty((volume.shape[0], volume.shape[1], volume.shape[2]))
        for i, vector_info in enumerate(vectors_with_points):
            start_point = vector_info[:3]
            vector_direction = vector_info[3:6]
            direction_norm = np.linalg.norm(vector_direction)
            if direction_norm == 0:
                continue
            normalized_direction = vector_direction / direction_norm
            step_vector = normalized_direction * (vector_length / num_steps)
            sum_values = 0
            neglect=0
            for step in range(num_steps + 1):
                point = start_point + step_vector * step
                interpolated_value = trilinear_interpolate(volume, *point)
                sum_values += interpolated_value
                
            mean_value = sum_values / (num_steps + 1-neglect)
            volume_2[int(start_point[0]), int(start_point[1]), int(start_point[2])] = mean_value
        return volume_2

    vector_length = 13
    num_steps = 20
    points_and_normals_sin = np.load(get_file_path("points_and_normals_sin_corrected.npy"))
    points_and_normals_dex = np.load(get_file_path("points_and_normals_dex_corrected.npy"))
    
    for i in range(2):
        if i == 0:
            points_and_normals = points_and_normals_sin
        else:
            points_and_normals = points_and_normals_dex
        means = mean_along_vectors(volume_brain, points_and_normals, vector_length, num_steps)
        brain_shell = means
        sigma = 0
        brain_shell_filtered = gaussian_filter(brain_shell, sigma=sigma)
        if i == 0:
            x0, y0, z0 = points_and_normals[:, 0], points_and_normals[:, 1], points_and_normals[:, 2]
            mask = np.zeros_like(brain_shell_filtered, dtype=bool)
            for x, y, z in zip(x0, y0, z0):
                mask[int(x), int(y), int(z)] = True
            pixel_values0 = brain_shell_filtered[mask]
        if i == 1:
            x1, y1, z1 = points_and_normals[:, 0], points_and_normals[:, 1], points_and_normals[:, 2]
            mask = np.zeros_like(brain_shell_filtered, dtype=bool)
            for x, y, z in zip(x1, y1, z1):
                mask[int(x), int(y), int(z)] = True
            pixel_values1 = brain_shell_filtered[mask]

    x2, y2, z2 = np.concatenate((x0, x1)), np.concatenate((y0, y1)), np.concatenate((z0, z1))
    pixel_values2 = np.concatenate((pixel_values0, pixel_values1))

    
    # Example image_array initialization
    image_array0 = np.zeros((128, 158, 128))
    image_array1 = np.zeros((128, 158, 128))
    image_array2 = np.zeros((128, 158, 128))


    # Assuming x2, y2, z2, and pixel_values2 are of the same length and contain valid indices
    for i in range(len(pixel_values0)):
        x, y, z = int(x0[i]), int(y0[i]), int(z0[i])  # Get the coordinates
        image_array0[x, y, z] = pixel_values0[i]
    # Replace all zeros with NaN
    image_array0[image_array0 == 0] = np.nan
    
    for i in range(len(pixel_values1)):
        x, y, z = int(x1[i]), int(y1[i]), int(z1[i])  # Get the coordinates
        image_array1[x, y, z] = pixel_values1[i]
    # Replace all zeros with NaN
    image_array1[image_array1 == 0] = np.nan
    
    for i in range(len(pixel_values2)):
        x, y, z = int(x2[i]), int(y2[i]), int(z2[i])  # Get the coordinates
        image_array2[x, y, z] = pixel_values2[i]
    # Replace all zeros with NaN
    image_array2[image_array2 == 0] = np.nan

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
    
    first_values_yz_pos_x0, first_values_yz_neg_x0, first_values_xz_pos_y0, first_values_xz_neg_y0= view_angels(image_array0)
    first_values_yz_pos_x1, first_values_yz_neg_x1, first_values_xz_pos_y1, first_values_xz_neg_y1= view_angels(image_array1)
    first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2= view_angels(image_array2)
    first_values_xz_neg_y2=np.flip(first_values_xz_neg_y2, axis=0)
    
    def plot_first_values(vmin=None, vmax=None):
        fig, axes = plt.subplots(1, 6, figsize=(20, 6))
        
        # Plot for the first non-NaN value along positive x-axis
        im1 = axes[0].imshow(np.rot90(np.rot90(np.rot90(first_values_yz_neg_x0))), interpolation='nearest', cmap=my_cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title('Dex inside')
        axes[0].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative x-axis
        im2 = axes[1].imshow(np.flip(np.rot90(np.rot90(np.rot90(first_values_yz_pos_x1))), axis=1), interpolation='nearest', cmap=my_cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title('Sin inside)')
        axes[1].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along positive y-axis
        im3 = axes[2].imshow(np.flip(np.rot90(np.rot90(np.rot90(first_values_yz_pos_x2))), axis=1), interpolation='nearest', cmap=my_cmap, vmin=vmin, vmax=vmax)
        axes[2].set_title('Dex')
        axes[2].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative y-axis
        im4 = axes[3].imshow(np.rot90(np.rot90(np.rot90(first_values_yz_neg_x2))), interpolation='nearest', cmap=my_cmap, vmin=vmin, vmax=vmax)
        axes[3].set_title('Sin')
        axes[3].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along positive y-axis
        im5 = axes[4].imshow(np.rot90(np.rot90(np.rot90(first_values_xz_pos_y2))), interpolation='nearest', cmap=my_cmap, vmin=vmin, vmax=vmax)
        axes[4].set_title('Back')
        axes[4].axis('off')  # Remove axis
    
        # Plot for the first non-NaN value along negative y-axis
        im6 = axes[5].imshow(np.rot90(np.rot90(np.rot90(first_values_xz_neg_y2))), interpolation='nearest', cmap=my_cmap, vmin=vmin, vmax=vmax)
        axes[5].set_title('Front')
        axes[5].axis('off')  # Remove axis
    
        # Add one shared colorbar for all subplots
        fig.subplots_adjust(right=0.85)  # Adjust to leave space for colorbar
        cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im1, cax=cbar_ax, label='ml/cm3/min')
    
        plt.tight_layout()
        plt.show()
    
    # Example usage:
    plot_first_values(vmin=0, vmax=1.2)
    # np.save('first_values_xz_neg_y2.npy', first_values_xz_neg_y2)
    return first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2

# transformed_K_1 = np.load("transformed_K_1.npy")
# SSP_2D(transformed_K_1)
