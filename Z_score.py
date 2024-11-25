# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:28:48 2024

@author: jacke
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import os
import sys
import numpy as np
import nibabel as nib

# Function to determine the correct file path
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
nifti_file_path = get_file_path(f'brain_masks.nii')  # Update this path if necessary


# Load the NIfTI file using nibabel
try:
    nifti_data = nii_gz_to_numpy(nifti_file_path)
    # # print(f"NIfTI file '{nifti_file_path}' loaded successfully.")
except FileNotFoundError as fnf_error:
    # # print(fnf_error)
    nifti_data = None
except IOError as io_error:
    # # print(io_error)
    nifti_data = None



def SD_corrected(transformed_K_1, ålder, kön, frame, wat):
    transformed_K_1[transformed_K_1 < 0.1] = np.nan
    background=np.load(get_file_path('background_down_corrected.npy'))
    
    grey = np.load(get_file_path('Register_grey_matter.npy'))
    grey = np.flip(grey, axis=1)
    grey = np.flip(grey, axis=2)
    grey = np.where(grey > 0.5, 1, np.nan)
    
    Cerebellum_dex=np.load(get_file_path('cerebellum_sin_down_corrected.npy'))*grey
    Cerebellum_dex[Cerebellum_dex < 0.5] =np.nan
    # Cerebellum_mean_dex=np.sum(transformed_K_1*Cerebellum_dex)/np.sum(Cerebellum_dex)
    Cerebellum_mean_dex=np.nanmean(transformed_K_1*Cerebellum_dex)
    Cerebellum_sin=np.load(get_file_path('cerebellum_dex_down_corrected.npy'))*grey
    Cerebellum_sin[Cerebellum_sin < 0.5] =np.nan
    # Cerebellum_mean_sin=np.sum(transformed_K_1*Cerebellum_sin)/np.sum(Cerebellum_sin)
    Cerebellum_mean_sin=np.nanmean(transformed_K_1*Cerebellum_sin)
    # # print('hej')
    # # print(Cerebellum_mean_dex)
    # # print(Cerebellum_mean_sin)
    nifti_data_dex = nifti_data.copy()
    nifti_data_sin = nifti_data.copy()
    nifti_data_dex[nifti_data<0.5]=np.nan
    nifti_data_sin[nifti_data<0.5]=np.nan
    
    Cerebellum_mean_dex_ref=np.nanmean(nifti_data_dex[..., 44]*transformed_K_1)
    Cerebellum_mean_sin_ref=np.nanmean(nifti_data_sin[..., 45]*transformed_K_1)
    Cerebellum_mean=(Cerebellum_mean_dex_ref+Cerebellum_mean_sin_ref)/2
    
    middle_dex=np.load(get_file_path('middle_sin_down_corrected.npy'))*grey
    middle_dex[middle_dex < 0.5] =np.nan
    # middle_mean_dex=np.sum(transformed_K_1*middle_dex)/np.sum(middle_dex)
    middle_mean_dex=np.nanmean(transformed_K_1*middle_dex)
    middle_sin=np.load(get_file_path('middle_dex_down_corrected.npy'))*grey
    middle_sin[middle_sin < 0.5] =np.nan
    # middle_mean_sin=np.sum(transformed_K_1*middle_sin)/np.sum(middle_sin)
    middle_mean_sin=np.nanmean(transformed_K_1*middle_sin)
    
    posterior_dex=np.load(get_file_path('posterior_sin_down_corrected.npy'))*grey
    posterior_dex[posterior_dex < 0.5] =np.nan
    # posterior_mean_dex=np.sum(transformed_K_1*posterior_dex)/np.sum(posterior_dex)
    posterior_mean_dex=np.nanmean(transformed_K_1*posterior_dex)
    posterior_sin=np.load(get_file_path('posterior_dex_down_corrected.npy'))*grey
    posterior_sin[posterior_sin < 0.5] =np.nan
    # posterior_mean_sin=np.sum(transformed_K_1*posterior_sin)/np.sum(posterior_sin)
    posterior_mean_sin=np.nanmean(transformed_K_1*posterior_sin)
    
    anterior_dex=np.load(get_file_path('anterior_sin_down_corrected.npy'))*grey
    anterior_dex[anterior_dex < 0.5] =np.nan
    # anterior_mean_dex=np.sum(transformed_K_1*anterior_dex)/np.sum(anterior_dex)
    anterior_mean_dex=np.nanmean(transformed_K_1*anterior_dex)
    anterior_sin=np.load(get_file_path('anterior_dex_down_corrected.npy'))*grey
    anterior_sin[anterior_sin < 0.5] =np.nan
    # anterior_mean_sin=np.sum(transformed_K_1*anterior_sin)/np.sum(anterior_sin)
    anterior_mean_sin=np.nanmean(transformed_K_1*anterior_sin)
    
    means_list=np.array([Cerebellum_mean_dex, Cerebellum_mean_sin, middle_mean_dex, middle_mean_sin, posterior_mean_dex, posterior_mean_sin, anterior_mean_dex, anterior_mean_sin])

    # plt.imshow(grey[:,:,100])
    # plt.show()
    # plt.imshow(Cerebellum_dex[:,:,100])
    # plt.show()
    # plt.imshow(transformed_K_1[:,:,100])
    # plt.show()
    
    
    
    
    # Let's read the data directly from the text file named "Mean perfusion with slope and intersect"
    if frame=="real" and wat== 'wat':
        file_path = get_file_path( 'flow_region_mean_std_k1.txt')
    elif  frame=="ref" and wat=='wat':
        file_path = get_file_path( 'flow_region_mean_std_k1_ref.txt')
    elif frame=='real' and wat=='wat1':
        file_path = get_file_path( 'flow_region_mean_std_k1_1.txt')
    elif frame=='ref' and wat=='wat1':
        file_path = get_file_path( 'flow_region_mean_std_k1_1_ref.txt')
    elif frame=='real' and wat=='wat2':
        file_path = get_file_path( 'flow_region_mean_std_k1_2.txt')
    elif frame=='ref' and wat=='wat2':
        file_path = get_file_path( 'flow_region_mean_std_k1_2_ref.txt')
    elif frame=='real' and wat=='wat1_2':
        file_path = get_file_path( 'flow_region_mean_std_k1_3.txt')
    elif frame=='ref' and wat=='wat1_2':
        file_path = get_file_path( 'flow_region_mean_std_k1_3_ref.txt')
    
    # Reading the text file and parsing it
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data_lines = [line.strip() for line in lines if not line.startswith('%') and line.strip()]

    # Skip the header row
    # data_lines = data_lines[1:]
    
    # Initialize lists for the columns
    regions, means, stds = [], [], []
    
    for line in data_lines:
        parts = line.split(',')
        regions.append(parts[0].strip())
        means.append(float(parts[1].strip()))
        stds.append(float(parts[2].strip()))
    
    # Create a DataFrame from the parsed data
    df_from_file = pd.DataFrame({
        'Region': regions,
        'Mean': means,
        'Std': stds,
        # 'Slope': slopes,
        # 'Intersect': intersects
    })

    # calculated_cerebellum_mean_dex=slopes[1]*ålder+intersects[1]
    z_cerebellum_dex=(Cerebellum_mean_dex-means[1])/stds[1]
    # calculated_cerebellum_mean_sin=slopes[0]*ålder+intersects[0]
    z_cerebellum_sin=(Cerebellum_mean_sin-means[0])/stds[0]

    # calculated_middle_mean_sin=slopes[3]*ålder+intersects[3]
    z_middle_dex=(middle_mean_dex-means[3])/stds[3]
    # calculated_middle_mean_sin=slopes[2]*ålder+intersects[2]
    z_middle_sin=(middle_mean_sin-means[2])/stds[2]

    # calculated_posterior_mean_dex=slopes[5]*ålder+intersects[5]
    z_posterior_dex=(posterior_mean_dex-means[5])/stds[5]
    # calculated_posterior_mean_sin=slopes[4]*ålder+intersects[4]
    z_posterior_sin=(posterior_mean_sin-means[4])/stds[4]

    # calculated_anterior_mean_dex=slopes[7]*ålder+intersects[7]
    z_anterior_dex=(anterior_mean_dex-means[7])/stds[7]
    # calculated_anterior_mean_sin=slopes[6]*ålder+intersects[6]
    z_anterior_sin=(anterior_mean_sin-means[6])/stds[6]
    
    
    
    # else:
    #     calculated_cerebellum_mean_dex=slopes[9]*ålder+intersects[9]
    #     z_cerebellum_dex=(Cerebellum_mean_dex-calculated_cerebellum_mean_dex)/stds[9]
    #     # print("Cerebellum dex: z =", (Cerebellum_mean_dex-calculated_cerebellum_mean_dex)/stds[9])
    #     calculated_cerebellum_mean_sin=slopes[8]*ålder+intersects[8]
    #     # print("Cerebellum sin: z =", (Cerebellum_mean_sin-calculated_cerebellum_mean_sin)/stds[8])
    
    #     calculated_middle_mean_dex=slopes[11]*ålder+intersects[11]
    #     # print("middle dex: z =", (calculated_middle_mean_dex-middle_mean_dex)/stds[11])
    #     calculated_middle_mean_sin=slopes[10]*ålder+intersects[10]
    #     # print("middle sin: z =", (calculated_middle_mean_sin-middle_mean_sin)/stds[10])
    
    #     calculated_posterior_mean_dex=slopes[13]*ålder+intersects[13]
    #     # print("posterior dex: z =", (calculated_posterior_mean_dex-posterior_mean_dex)/stds[13])
    #     calculated_posterior_mean_sin=slopes[12]*ålder+intersects[12]
    #     # print("posterior sin: z =", (calculated_posterior_mean_sin-posterior_mean_sin)/stds[12])
    
    #     calculated_anterior_mean_dex=slopes[15]*ålder+intersects[15]
    #     # print("anterior dex: z =", (calculated_anterior_mean_dex-anterior_mean_dex)/stds[15])
    #     calculated_anterior_mean_sin=slopes[14]*ålder+intersects[14]
    #     # print("anterior sin: z =", (calculated_anterior_mean_sin-anterior_mean_sin)/stds[14])
        
    cerebellum_sin_all=np.load(get_file_path('cerebellum_dex_down_corrected.npy') )   
    cerebellum_dex_all=np.load(get_file_path('cerebellum_sin_down_corrected.npy') )
    middle_sin_all=np.load(get_file_path('middle_dex_down_corrected.npy') )
    middle_dex_all=np.load(get_file_path('middle_sin_down_corrected.npy') )
    posterior_sin_all=np.load(get_file_path('posterior_dex_down_corrected.npy') )
    posterior_dex_all=np.load(get_file_path('posterior_sin_down_corrected.npy') )
    anterior_sin_all=np.load(get_file_path('anterior_dex_down_corrected.npy') )
    anterior_dex_all=np.load(get_file_path('anterior_sin_down_corrected.npy') )
    
    Z_brain=z_cerebellum_dex*cerebellum_dex_all+z_cerebellum_sin*cerebellum_sin_all\
        +z_anterior_dex*anterior_dex_all+z_anterior_sin*anterior_sin_all\
        +z_middle_dex*middle_dex_all+z_middle_sin*middle_sin_all\
        +z_posterior_dex*posterior_dex_all+z_posterior_sin*posterior_sin_all\
        +background
    
    z_scores_flow=[z_cerebellum_dex, z_cerebellum_sin, z_anterior_dex, z_anterior_sin, z_middle_dex,
                   z_middle_sin, z_posterior_dex, z_posterior_sin]
    
    return Z_brain, z_scores_flow, Cerebellum_mean, means_list


def SD_corrected_park(R_I_reshape_list, tracer):
    
    # R_I_reshape_list=np.where(R_I_reshape_list >=0.1, R_I_reshape_list, np.nan)
    R_I_reshape_list[R_I_reshape_list<0.1]=np.nan
    
    # R_I_mask=np.where(R_I_reshape_list >=0.1, 1, np.nan)
    
    
    background=np.load(get_file_path('background_down_corrected.npy'))
    
    grey = np.load(get_file_path('Register_grey_matter.npy'))
    grey = np.flip(grey, axis=1)
    grey = np.flip(grey, axis=2)
    grey = np.where(grey > 0.5, 1, 0)
    
    Cerebellum_dex=np.load(get_file_path('cerebellum_dex_down_corrected.npy'))
    # Cerebellum_mean_dex=np.nansum(R_I_reshape_list*Cerebellum_dex)/np.nansum(Cerebellum_dex*R_I_mask)
    Cerebellum_dex[Cerebellum_dex < 0.5] =np.nan
    Cerebellum_mean_dex=np.nanmean(R_I_reshape_list*Cerebellum_dex)
    Cerebellum_sin=np.load(get_file_path('cerebellum_sin_down_corrected.npy'))
    # Cerebellum_mean_sin=np.nansum(R_I_reshape_list*Cerebellum_sin)/np.nansum(Cerebellum_sin*R_I_mask)
    Cerebellum_sin[Cerebellum_sin < 0.5] =np.nan
    Cerebellum_mean_sin=np.nanmean(R_I_reshape_list*Cerebellum_sin)
    
    middle_dex=np.load(get_file_path('middle_dex_down_corrected.npy'))
    # middle_mean_dex=np.nansum(R_I_reshape_list*middle_dex)/np.nansum(middle_dex*R_I_mask)
    middle_dex[middle_dex < 0.5] =np.nan
    middle_mean_dex=np.nanmean(R_I_reshape_list*middle_dex)
    middle_sin=np.load(get_file_path('middle_sin_down_corrected.npy'))
    # middle_mean_sin=np.nansum(R_I_reshape_list*middle_sin)/np.nansum(middle_sin*R_I_mask)
    middle_sin[middle_sin < 0.5] =np.nan
    middle_mean_sin=np.nanmean(R_I_reshape_list*middle_sin)
    
    posterior_dex=np.load(get_file_path('posterior_dex_down_corrected.npy'))
    posterior_dex[posterior_dex < 0.5] =np.nan
    posterior_mean_dex=np.nanmean(R_I_reshape_list*posterior_dex)
    # posterior_mean_dex=np.nansum(R_I_reshape_list*posterior_dex)/np.nansum(posterior_dex*R_I_mask)
    posterior_sin=np.load(get_file_path('posterior_sin_down_corrected.npy'))
    # posterior_mean_sin=np.nansum(R_I_reshape_list*posterior_sin)/np.nansum(posterior_sin*R_I_mask)
    posterior_sin[posterior_sin < 0.5] =np.nan
    posterior_mean_sin=np.nanmean(R_I_reshape_list*posterior_sin)
    
    anterior_dex=np.load(get_file_path('anterior_dex_down_corrected.npy'))
    # anterior_mean_dex=np.nansum(R_I_reshape_list*anterior_dex)/np.nansum(anterior_dex*R_I_mask)
    anterior_dex[anterior_dex < 0.5] =np.nan
    anterior_mean_dex=np.nanmean(R_I_reshape_list*anterior_dex)
    anterior_sin=np.load(get_file_path('anterior_sin_down_corrected.npy'))
    # anterior_mean_sin=np.nansum(R_I_reshape_list*anterior_sin)/np.nansum(anterior_sin*R_I_mask)
    anterior_sin[anterior_sin < 0.5] =np.nan
    anterior_mean_sin=np.nanmean(R_I_reshape_list*anterior_sin)
    
    means_list=np.array([Cerebellum_mean_dex, Cerebellum_mean_sin, middle_mean_dex, middle_mean_sin, posterior_mean_dex, posterior_mean_sin, anterior_mean_dex, anterior_mean_sin])

    
    # plt.imshow(grey[:,:,100])
    # plt.show()
    # plt.imshow(Cerebellum_dex[:,:,100])
    # plt.show()
    # plt.imshow(R_I_reshape_list[:,:,100])
    # plt.show()
    
    
    
    
    # Let's read the data directly from the text file named "Mean perfusion with slope and intersect"
    if tracer == "C-11":
        file_path = get_file_path('48_region_mean_std_R_I_C.txt')
    else:
        file_path = get_file_path('48_region_mean_std_R_I.txt')
    
    # Reading the text file and parsing it
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract the relevant lines, removing headers and footer sections
    data_lines = [line.strip() for line in lines if not line.startswith('%') and line.strip()]
    
    # Now let's split the values and create a list for each relevant column
    regions, means, stds = [], [], []
    
    for line in data_lines:
        if ':' in line:
            # This is the label "Male:" which we don't need to include
            continue
        parts = line.split(',')
        regions.append(parts[0].strip())
        means.append(float(parts[1].strip()))
        stds.append(float(parts[2].strip()))

    
    # Create a DataFrame from the parsed data
    df_from_file = pd.DataFrame({
        'Region': regions,
        'Mean': means,
        'Std': stds,
    })

    # calculated_cerebellum_mean_dex=slopes[1]*ålder+intersects[1]
    z_cerebellum_dex=(Cerebellum_mean_dex-means[1])/stds[1]
    # calculated_cerebellum_mean_sin=slopes[0]*ålder+intersects[0]
    z_cerebellum_sin=(Cerebellum_mean_sin-means[0])/stds[0]

    # calculated_middle_mean_sin=slopes[3]*ålder+intersects[3]
    z_middle_dex=(middle_mean_dex-means[3])/stds[3]
    # calculated_middle_mean_sin=slopes[2]*ålder+intersects[2]
    z_middle_sin=(middle_mean_sin-means[2])/stds[2]

    # calculated_posterior_mean_dex=slopes[5]*ålder+intersects[5]
    z_posterior_dex=(posterior_mean_dex-means[5])/stds[5]
    # calculated_posterior_mean_sin=slopes[4]*ålder+intersects[4]
    z_posterior_sin=(posterior_mean_sin-means[4])/stds[4]

    # calculated_anterior_mean_dex=slopes[7]*ålder+intersects[7]
    z_anterior_dex=(anterior_mean_dex-means[7])/stds[7]
    # calculated_anterior_mean_sin=slopes[6]*ålder+intersects[6]
    z_anterior_sin=(anterior_mean_sin-means[6])/stds[6]
        
    
        
    cerebellum_dex_all=np.load(get_file_path('cerebellum_dex_down_corrected.npy')    )
    cerebellum_sin_all=np.load(get_file_path('cerebellum_sin_down_corrected.npy') )
    middle_dex_all=np.load(get_file_path('middle_dex_down_corrected.npy') )
    middle_sin_all=np.load(get_file_path('middle_sin_down_corrected.npy') )
    posterior_dex_all=np.load(get_file_path('posterior_dex_down_corrected.npy') )
    posterior_sin_all=np.load(get_file_path('posterior_sin_down_corrected.npy') )
    anterior_dex_all=np.load(get_file_path('anterior_dex_down_corrected.npy') )
    anterior_sin_all=np.load(get_file_path('anterior_sin_down_corrected.npy') )
    
    Z_brain=z_cerebellum_dex*cerebellum_dex_all+z_cerebellum_sin*cerebellum_sin_all\
        +z_anterior_dex*anterior_dex_all+z_anterior_sin*anterior_sin_all\
        +z_middle_dex*middle_dex_all+z_middle_sin*middle_sin_all\
        +z_posterior_dex*posterior_dex_all+z_posterior_sin*posterior_sin_all\
        +background
    
    z_scores_flow=[z_cerebellum_dex, z_cerebellum_sin, z_anterior_dex, z_anterior_sin, z_middle_dex,
                   z_middle_sin, z_posterior_dex, z_posterior_sin]
    return Z_brain, z_scores_flow, means_list

