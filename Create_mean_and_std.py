# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:27:06 2024

@author: jacke
"""

import zipfile
import numpy as np
from io import BytesIO
from pathlib import Path
import nibabel as nib
import os
import pandas as pd  # Import pandas for handling tabular data
import pickle
import numpy as np
import zipfile
import nibabel as nib
import os
import tempfile
import pickle
import matplotlib.pyplot as plt
import time
import sys
import os
import tempfile
import traceback
import shutil
import sys

# Create a temporary file
temp_output = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.log')
temp_output_path = temp_output.name

# Redirect stdout and stderr
sys.stdout = temp_output
sys.stderr = temp_output


# Function to determine the correct file path
def get_file_path(filename):
    if hasattr(sys, '_MEIPASS'):
        # If running in a PyInstaller bundle, look in the temporary directory
        return os.path.join(sys._MEIPASS, filename)
    else:
        # Otherwise, look in the current working directory
        return os.path.join(os.getcwd(), filename)

def load_nifti_data(input_path):
    """
    Loads NIfTI files from a ZIP archive or a regular directory into a NumPy array.
    
    Parameters:
    - input_path (str): Path to the ZIP file or directory containing .nii or .nii.gz files.
    
    Returns:
    - data_array (np.ndarray): 4D array containing all loaded NIfTI data.
    """
    def get_nifti_files_from_directory(directory):
        nii_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.nii') or file.lower().endswith('.nii.gz'):
                    nii_files.append(os.path.join(root, file))
        return nii_files

    temp_dir = None  # Initialize temp_dir to None

    try:
        # If input is a ZIP file, extract to a fixed temporary directory
        if os.path.isfile(input_path) and zipfile.is_zipfile(input_path):
            temp_dir = "C:/Temp/nifti_extraction"
            os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                extracted_files = zip_ref.namelist()
                # print(f"Extracted files: {extracted_files}")  # Debugging output
            
            nii_files = get_nifti_files_from_directory(temp_dir)
        elif os.path.isdir(input_path):
            # If input is a directory, directly list NIfTI files
            nii_files = get_nifti_files_from_directory(input_path)
        else:
            raise ValueError("Input path must be a ZIP file or a directory.")

        if not nii_files:
            raise ValueError("No .nii or .nii.gz files found in the specified path.")

        # print(f"Found {len(nii_files)} NIfTI files in the input path.")

        # Load the first NIfTI file to get the shape
        first_img = nib.load(nii_files[0])
        first_data = first_img.get_fdata()
        x, y, z = first_data.shape

        # Initialize a NumPy array to hold all data
        num_files = len(nii_files)
        data_array = np.zeros((x, y, z, num_files), dtype=first_data.dtype)

        # Load each NIfTI file and assign to the NumPy array
        for idx, nii_file in enumerate(nii_files):
            img = nib.load(nii_file)
            data = img.get_fdata()

            # Verify that all files have the same shape
            if data.shape != (x, y, z):
                raise ValueError(f"Shape mismatch in file {nii_file}: expected {(x, y, z)}, got {data.shape}")

            data_array[..., idx] = data
            # print(f"Loaded file {idx+1}/{num_files}: {nii_file}")

        # print(f"Successfully loaded {num_files} NIfTI files into a NumPy array with shape {data_array.shape}")
        return data_array

    finally:
        # Clean up the temporary directory if it was used
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            # print(f"Temporary directory {temp_dir} has been removed.")





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

regions = [
    "Cerebellum sin", "Cerebellum dex", 
    "Middle sin", "Middle dex", 
    "Posterior sin", "Posterior dex", 
    "Anterior sin", "Anterior dex"
]

# Define the path to the ZIP file
# zip_file_path = r"C:\Users\jacke\OneDrive\Skrivbord\K_1_nii.zip"



# Path to the NIfTI file containing all masks
nifti_file_path = get_file_path('brain_masks.nii')  # Update this path if necessary


# Load the NIfTI file using nibabel
try:
    nifti_data = nii_gz_to_numpy(nifti_file_path)
    # print(f"NIfTI file '{nifti_file_path}' loaded successfully.")
except FileNotFoundError as fnf_error:
    # # print(fnf_error)
    nifti_data = None
except IOError as io_error:
    # # print(io_error)
    nifti_data = None
    
start_tid=time.time()

out_path=r"C:\Users\jacke\OneDrive\Skrivbord"




def water_mean_std(path, out_path): 
    # # print('kör water med', path)
    data_array = load_nifti_data(path)

    data_array=np.rot90(data_array, k=2, axes=(1,2))
    data_array=np.rot90(data_array, k=2, axes=(0,1))
    
    flow_mean_list_all_1=[]
    
    first_values_yz_neg_x0_list_k1_1=[]
    first_values_yz_pos_x1_list_k1_1=[]
    first_values_yz_pos_x2_list_k1_1=[]
    first_values_yz_neg_x2_list_k1_1=[]
    first_values_xz_pos_y2_list_k1_1=[]
    first_values_xz_neg_y2_list_k1_1=[]
    
    region_mean_values_1=[]
    
    flow_mean_list_all_ref_1=[]
    
    first_values_yz_neg_x0_list_k1_1_ref=[]
    first_values_yz_pos_x1_list_k1_1_ref=[]
    first_values_yz_pos_x2_list_k1_1_ref=[]
    first_values_yz_neg_x2_list_k1_1_ref=[]
    first_values_xz_pos_y2_list_k1_1_ref=[]
    first_values_xz_neg_y2_list_k1_1_ref=[]
    
    region_mean_values_ref_1=[]
    
    for i in range(np.shape(data_array)[3]):
           
        from SSP_2d import SSP_2D
        first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2=SSP_2D(data_array[..., i])
        first_values_yz_neg_x0_list_k1_1.append(first_values_yz_neg_x0)
        first_values_yz_pos_x1_list_k1_1.append(first_values_yz_pos_x1)
        first_values_yz_pos_x2_list_k1_1.append(first_values_yz_pos_x2)
        first_values_yz_neg_x2_list_k1_1.append(first_values_yz_neg_x2)
        first_values_xz_pos_y2_list_k1_1.append(first_values_xz_pos_y2)
        first_values_xz_neg_y2_list_k1_1.append(first_values_xz_neg_y2)
        
        from Z_score import SD_corrected
        ålder=1 
        kön='M'
        frame='real'
        wat='wat1'
        igon, igon, Cerebellum_mean_k1, flow_mean_list =SD_corrected(data_array[..., i], ålder, kön, frame, wat)
        flow_mean_list_all_1.append(flow_mean_list)
        # # # print(flow_mean_list)
        
        from plot_regioner_mean import regions_z_score
        
        igno, igno, igno, igno, igno, igno, igno, igno, mean_values=regions_z_score(data_array[..., i], frame, wat)
        
        #mask_names is the key to the mean values    
        region_mean_values_temp=[]
        # # print('i=', i)
        for name in mask_names:
            region_mean_values_temp.append(mean_values[name])
            # # # print(f'{name}: mean= {mean_values[name]}')
        region_mean_values_1.append(region_mean_values_temp)
        
        igon, igno, Cerebellum_mean_k1_ref, flow_mean_list_ref =SD_corrected(data_array[..., i]/Cerebellum_mean_k1, ålder, kön, frame, wat)
        flow_mean_list_all_ref_1.append(flow_mean_list_ref)
        
        
        #reference
        
        igno, igno, igno, igno, igno, igno, igon, igon, mean_values_ref = regions_z_score(data_array[..., i]/Cerebellum_mean_k1, 'ref', 'wat1')
        
        #mask_names is the key to the mean values    
        region_mean_values_temp_ref=[]
        # # print('i=', i)
        for name in mask_names:
            region_mean_values_temp_ref.append(mean_values_ref[name])
            # # # print(f'{name}: mean= {mean_values[name]}')
        region_mean_values_ref_1.append(region_mean_values_temp_ref)
    
        
        first_values_yz_neg_x0_ref, first_values_yz_pos_x1_ref, first_values_yz_pos_x2_ref, first_values_yz_neg_x2_ref, first_values_xz_pos_y2_ref, first_values_xz_neg_y2_ref=first_values_yz_neg_x0/Cerebellum_mean_k1, first_values_yz_pos_x1/Cerebellum_mean_k1, first_values_yz_pos_x2/Cerebellum_mean_k1, first_values_yz_neg_x2/Cerebellum_mean_k1, first_values_xz_pos_y2/Cerebellum_mean_k1, first_values_xz_neg_y2/Cerebellum_mean_k1
        first_values_yz_neg_x0_list_k1_1_ref.append(first_values_yz_neg_x0_ref)
        first_values_yz_pos_x1_list_k1_1_ref.append(first_values_yz_pos_x1_ref)
        first_values_yz_pos_x2_list_k1_1_ref.append(first_values_yz_pos_x2_ref)
        first_values_yz_neg_x2_list_k1_1_ref.append(first_values_yz_neg_x2_ref)
        first_values_xz_pos_y2_list_k1_1_ref.append(first_values_xz_pos_y2_ref)
        first_values_xz_neg_y2_list_k1_1_ref.append(first_values_xz_neg_y2_ref)
        
    
        
    # # print('total tid:', time.time()-start_tid)
    
    flow_mean_list_all_1=np.array(flow_mean_list_all_1)
    flow_mean_k1_1=np.mean(flow_mean_list_all_1, axis=0)
    flow_std_k1_1=np.std(flow_mean_list_all_1, axis=0)
    
    
    first_values_yz_neg_x0_mean_k1_1=np.mean(first_values_yz_neg_x0_list_k1_1, axis=0)
    first_values_yz_pos_x1_mean_k1_1=np.mean(first_values_yz_pos_x1_list_k1_1, axis=0)
    first_values_yz_pos_x2_mean_k1_1=np.mean(first_values_yz_pos_x2_list_k1_1, axis=0)
    first_values_yz_neg_x2_mean_k1_1=np.mean(first_values_yz_neg_x2_list_k1_1, axis=0)
    first_values_xz_pos_y2_mean_k1_1=np.mean(first_values_xz_pos_y2_list_k1_1, axis=0)
    first_values_xz_neg_y2_mean_k1_1=np.mean(first_values_xz_neg_y2_list_k1_1, axis=0)
    
    first_values_yz_neg_x0_std_k1_1=np.std(first_values_yz_neg_x0_list_k1_1, axis=0)
    first_values_yz_pos_x1_std_k1_1=np.std(first_values_yz_pos_x1_list_k1_1, axis=0)
    first_values_yz_pos_x2_std_k1_1=np.std(first_values_yz_pos_x2_list_k1_1, axis=0)
    first_values_yz_neg_x2_std_k1_1=np.std(first_values_yz_neg_x2_list_k1_1, axis=0)
    first_values_xz_pos_y2_std_k1_1=np.std(first_values_xz_pos_y2_list_k1_1, axis=0)
    first_values_xz_neg_y2_std_k1_1=np.std(first_values_xz_neg_y2_list_k1_1, axis=0)
    
    
    region_mean_values_1=np.array(region_mean_values_1)
    region_mean_k1_1=np.mean(region_mean_values_1, axis=0)
    region_std_k1_1=np.std(region_mean_values_1, axis=0)
    
    
    
    
    flow_mean_list_all_ref_1=np.array(flow_mean_list_all_ref_1)
    flow_mean_k1_1_ref=np.mean(flow_mean_list_all_ref_1, axis=0)
    flow_std_k1_1_ref=np.std(flow_mean_list_all_ref_1, axis=0)
    
    first_values_yz_neg_x0_mean_k1_1_ref=np.mean(first_values_yz_neg_x0_list_k1_1_ref, axis=0)
    first_values_yz_pos_x1_mean_k1_1_ref=np.mean(first_values_yz_pos_x1_list_k1_1_ref, axis=0)
    first_values_yz_pos_x2_mean_k1_1_ref=np.mean(first_values_yz_pos_x2_list_k1_1_ref, axis=0)
    first_values_yz_neg_x2_mean_k1_1_ref=np.mean(first_values_yz_neg_x2_list_k1_1_ref, axis=0)
    first_values_xz_pos_y2_mean_k1_1_ref=np.mean(first_values_xz_pos_y2_list_k1_1_ref, axis=0)
    first_values_xz_neg_y2_mean_k1_1_ref=np.mean(first_values_xz_neg_y2_list_k1_1_ref, axis=0)
    
    first_values_yz_neg_x0_std_k1_1_ref=np.std(first_values_yz_neg_x0_list_k1_1_ref, axis=0)
    first_values_yz_pos_x1_std_k1_1_ref=np.std(first_values_yz_pos_x1_list_k1_1_ref, axis=0)
    first_values_yz_pos_x2_std_k1_1_ref=np.std(first_values_yz_pos_x2_list_k1_1_ref, axis=0)
    first_values_yz_neg_x2_std_k1_1_ref=np.std(first_values_yz_neg_x2_list_k1_1_ref, axis=0)
    first_values_xz_pos_y2_std_k1_1_ref=np.std(first_values_xz_pos_y2_list_k1_1_ref, axis=0)
    first_values_xz_neg_y2_std_k1_1_ref=np.std(first_values_xz_neg_y2_list_k1_1_ref, axis=0)
    
    region_mean_values_ref_1=np.array(region_mean_values_ref_1)
    region_mean_k1_1_ref=np.mean(region_mean_values_ref_1, axis=0)
    region_std_k1_1_ref=np.std(region_mean_values_ref_1, axis=0)
    
    
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(regions, flow_mean_k1_1, flow_std_k1_1)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "flow_region_mean_std_k1.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
        
    mean_and_std_first_values_yz_neg_x0_k1_1=[first_values_yz_neg_x0_mean_k1_1, first_values_yz_neg_x0_std_k1_1]
    mean_and_std_first_values_yz_pos_x1_k1_1=[first_values_yz_pos_x1_mean_k1_1, first_values_yz_pos_x1_std_k1_1]
    mean_and_std_first_values_yz_pos_x2_k1_1=[first_values_yz_pos_x2_mean_k1_1, first_values_yz_pos_x2_std_k1_1]
    mean_and_std_first_values_yz_neg_x2_k1_1=[first_values_yz_neg_x2_mean_k1_1, first_values_yz_neg_x2_std_k1_1]
    mean_and_std_first_values_xz_pos_y2_k1_1=[first_values_xz_pos_y2_mean_k1_1, first_values_xz_pos_y2_std_k1_1]
    mean_and_std_first_values_xz_neg_y2_k1_1=[first_values_xz_neg_y2_mean_k1_1, first_values_xz_neg_y2_std_k1_1]
    
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x0_k1.npy'), mean_and_std_first_values_yz_neg_x0_k1_1)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x1_k1.npy'), mean_and_std_first_values_yz_pos_x1_k1_1)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x2_k1.npy'), mean_and_std_first_values_yz_pos_x2_k1_1)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x2_k1.npy'), mean_and_std_first_values_yz_neg_x2_k1_1)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_pos_y2_k1.npy'), mean_and_std_first_values_xz_pos_y2_k1_1)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_neg_y2_k1.npy'), mean_and_std_first_values_xz_neg_y2_k1_1)
    
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(mask_names, region_mean_k1_1, region_std_k1_1)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "48_region_mean_std_k1.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
    
    #reference:
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(regions, flow_mean_k1_1_ref, flow_std_k1_1_ref)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "flow_region_mean_std_k1_ref.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
        
    mean_and_std_first_values_yz_neg_x0_k1_1_ref=[first_values_yz_neg_x0_mean_k1_1_ref, first_values_yz_neg_x0_std_k1_1_ref]
    mean_and_std_first_values_yz_pos_x1_k1_1_ref=[first_values_yz_pos_x1_mean_k1_1_ref, first_values_yz_pos_x1_std_k1_1_ref]
    mean_and_std_first_values_yz_pos_x2_k1_1_ref=[first_values_yz_pos_x2_mean_k1_1_ref, first_values_yz_pos_x2_std_k1_1_ref]
    mean_and_std_first_values_yz_neg_x2_k1_1_ref=[first_values_yz_neg_x2_mean_k1_1_ref, first_values_yz_neg_x2_std_k1_1_ref]
    mean_and_std_first_values_xz_pos_y2_k1_1_ref=[first_values_xz_pos_y2_mean_k1_1_ref, first_values_xz_pos_y2_std_k1_1_ref]
    mean_and_std_first_values_xz_neg_y2_k1_1_ref=[first_values_xz_neg_y2_mean_k1_1_ref, first_values_xz_neg_y2_std_k1_1_ref]
    
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x0_k1_ref.npy'), mean_and_std_first_values_yz_neg_x0_k1_1_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x1_k1_ref.npy'), mean_and_std_first_values_yz_pos_x1_k1_1_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x2_k1_ref.npy'), mean_and_std_first_values_yz_pos_x2_k1_1_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x2_k1_ref.npy'), mean_and_std_first_values_yz_neg_x2_k1_1_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_pos_y2_k1_ref.npy'), mean_and_std_first_values_xz_pos_y2_k1_1_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_neg_y2_k1_ref.npy'), mean_and_std_first_values_xz_neg_y2_k1_1_ref)
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(mask_names, region_mean_k1_1_ref, region_std_k1_1_ref)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "48_region_mean_std_k1_ref.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
# path = r"C:\Users\jacke\OneDrive\Skrivbord\K_1_nii.zip"   
# water_mean_std(path, out_path)

#-------------------------------------------------------------------------------------------------
#wat1
#-------------------------------------------------------------------------------------------------
def wat1_wat2_mean_std(path_1, path_2, out_path): 
    # # print('kör wat2 wat1 med', path_1, path_2)
    data_array_1 = load_nifti_data(path_1)
    # # print('data_array', data_array_1.shape)
    
    data_array_1=np.rot90(data_array_1, k=2, axes=(1,2))
    data_array_1=np.rot90(data_array_1, k=2, axes=(0,1))
    
    flow_mean_list_all_1=[]
    
    first_values_yz_neg_x0_list_k1_1=[]
    first_values_yz_pos_x1_list_k1_1=[]
    first_values_yz_pos_x2_list_k1_1=[]
    first_values_yz_neg_x2_list_k1_1=[]
    first_values_xz_pos_y2_list_k1_1=[]
    first_values_xz_neg_y2_list_k1_1=[]
    
    region_mean_values_1=[]
    
    flow_mean_list_all_ref_1=[]
    
    first_values_yz_neg_x0_list_k1_1_ref=[]
    first_values_yz_pos_x1_list_k1_1_ref=[]
    first_values_yz_pos_x2_list_k1_1_ref=[]
    first_values_yz_neg_x2_list_k1_1_ref=[]
    first_values_xz_pos_y2_list_k1_1_ref=[]
    first_values_xz_neg_y2_list_k1_1_ref=[]
    
    region_mean_values_ref_1=[]
    
    for i in range(np.shape(data_array_1)[3]):
           
        from SSP_2d import SSP_2D
        first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2=SSP_2D(data_array_1[..., i])
        first_values_yz_neg_x0_list_k1_1.append(first_values_yz_neg_x0)
        first_values_yz_pos_x1_list_k1_1.append(first_values_yz_pos_x1)
        first_values_yz_pos_x2_list_k1_1.append(first_values_yz_pos_x2)
        first_values_yz_neg_x2_list_k1_1.append(first_values_yz_neg_x2)
        first_values_xz_pos_y2_list_k1_1.append(first_values_xz_pos_y2)
        first_values_xz_neg_y2_list_k1_1.append(first_values_xz_neg_y2)
        
        from Z_score import SD_corrected
        ålder=1 
        kön='M'
        frame='real'
        wat='wat1'
        igon, igon, Cerebellum_mean_k1, flow_mean_list =SD_corrected(data_array_1[..., i], ålder, kön, frame, wat)
        flow_mean_list_all_1.append(flow_mean_list)
        # # # print(flow_mean_list)
        
        from plot_regioner_mean import regions_z_score
        
        igno, igno, igno, igno, igno, igno, igno, igno, mean_values=regions_z_score(data_array_1[..., i], frame, wat)
        
        #mask_names is the key to the mean values    
        region_mean_values_temp=[]
        # # print('i=', i)
        for name in mask_names:
            region_mean_values_temp.append(mean_values[name])
            # # # print(f'{name}: mean= {mean_values[name]}')
        region_mean_values_1.append(region_mean_values_temp)
        
        igon, igno, Cerebellum_mean_k1_ref, flow_mean_list_ref =SD_corrected(data_array_1[..., i]/Cerebellum_mean_k1, ålder, kön, frame, wat)
        flow_mean_list_all_ref_1.append(flow_mean_list_ref)
        
        
        #reference
        
        igno, igno, igno, igno, igno, igno, igon, igon, mean_values_ref = regions_z_score(data_array_1[..., i]/Cerebellum_mean_k1, 'ref', 'wat1')
        
        #mask_names is the key to the mean values    
        region_mean_values_temp_ref=[]
        # # print('i=', i)
        for name in mask_names:
            region_mean_values_temp_ref.append(mean_values_ref[name])
            # # print(f'{name}: mean= {mean_values[name]}')
        region_mean_values_ref_1.append(region_mean_values_temp_ref)
    
        
        first_values_yz_neg_x0_ref, first_values_yz_pos_x1_ref, first_values_yz_pos_x2_ref, first_values_yz_neg_x2_ref, first_values_xz_pos_y2_ref, first_values_xz_neg_y2_ref=first_values_yz_neg_x0/Cerebellum_mean_k1, first_values_yz_pos_x1/Cerebellum_mean_k1, first_values_yz_pos_x2/Cerebellum_mean_k1, first_values_yz_neg_x2/Cerebellum_mean_k1, first_values_xz_pos_y2/Cerebellum_mean_k1, first_values_xz_neg_y2/Cerebellum_mean_k1
        first_values_yz_neg_x0_list_k1_1_ref.append(first_values_yz_neg_x0_ref)
        first_values_yz_pos_x1_list_k1_1_ref.append(first_values_yz_pos_x1_ref)
        first_values_yz_pos_x2_list_k1_1_ref.append(first_values_yz_pos_x2_ref)
        first_values_yz_neg_x2_list_k1_1_ref.append(first_values_yz_neg_x2_ref)
        first_values_xz_pos_y2_list_k1_1_ref.append(first_values_xz_pos_y2_ref)
        first_values_xz_neg_y2_list_k1_1_ref.append(first_values_xz_neg_y2_ref)
        
    
        
    # print('total tid:', time.time()-start_tid)
    
    flow_mean_list_all_1=np.array(flow_mean_list_all_1)
    flow_mean_k1_1=np.mean(flow_mean_list_all_1, axis=0)
    flow_std_k1_1=np.std(flow_mean_list_all_1, axis=0)
    
    
    first_values_yz_neg_x0_mean_k1_1=np.mean(first_values_yz_neg_x0_list_k1_1, axis=0)
    first_values_yz_pos_x1_mean_k1_1=np.mean(first_values_yz_pos_x1_list_k1_1, axis=0)
    first_values_yz_pos_x2_mean_k1_1=np.mean(first_values_yz_pos_x2_list_k1_1, axis=0)
    first_values_yz_neg_x2_mean_k1_1=np.mean(first_values_yz_neg_x2_list_k1_1, axis=0)
    first_values_xz_pos_y2_mean_k1_1=np.mean(first_values_xz_pos_y2_list_k1_1, axis=0)
    first_values_xz_neg_y2_mean_k1_1=np.mean(first_values_xz_neg_y2_list_k1_1, axis=0)
    
    first_values_yz_neg_x0_std_k1_1=np.std(first_values_yz_neg_x0_list_k1_1, axis=0)
    first_values_yz_pos_x1_std_k1_1=np.std(first_values_yz_pos_x1_list_k1_1, axis=0)
    first_values_yz_pos_x2_std_k1_1=np.std(first_values_yz_pos_x2_list_k1_1, axis=0)
    first_values_yz_neg_x2_std_k1_1=np.std(first_values_yz_neg_x2_list_k1_1, axis=0)
    first_values_xz_pos_y2_std_k1_1=np.std(first_values_xz_pos_y2_list_k1_1, axis=0)
    first_values_xz_neg_y2_std_k1_1=np.std(first_values_xz_neg_y2_list_k1_1, axis=0)
    
    
    region_mean_values_1=np.array(region_mean_values_1)
    region_mean_k1_1=np.mean(region_mean_values_1, axis=0)
    region_std_k1_1=np.std(region_mean_values_1, axis=0)
    
    
    
    
    flow_mean_list_all_ref_1=np.array(flow_mean_list_all_ref_1)
    flow_mean_k1_1_ref=np.mean(flow_mean_list_all_ref_1, axis=0)
    flow_std_k1_1_ref=np.std(flow_mean_list_all_ref_1, axis=0)
    
    first_values_yz_neg_x0_mean_k1_1_ref=np.mean(first_values_yz_neg_x0_list_k1_1_ref, axis=0)
    first_values_yz_pos_x1_mean_k1_1_ref=np.mean(first_values_yz_pos_x1_list_k1_1_ref, axis=0)
    first_values_yz_pos_x2_mean_k1_1_ref=np.mean(first_values_yz_pos_x2_list_k1_1_ref, axis=0)
    first_values_yz_neg_x2_mean_k1_1_ref=np.mean(first_values_yz_neg_x2_list_k1_1_ref, axis=0)
    first_values_xz_pos_y2_mean_k1_1_ref=np.mean(first_values_xz_pos_y2_list_k1_1_ref, axis=0)
    first_values_xz_neg_y2_mean_k1_1_ref=np.mean(first_values_xz_neg_y2_list_k1_1_ref, axis=0)
    
    first_values_yz_neg_x0_std_k1_1_ref=np.std(first_values_yz_neg_x0_list_k1_1_ref, axis=0)
    first_values_yz_pos_x1_std_k1_1_ref=np.std(first_values_yz_pos_x1_list_k1_1_ref, axis=0)
    first_values_yz_pos_x2_std_k1_1_ref=np.std(first_values_yz_pos_x2_list_k1_1_ref, axis=0)
    first_values_yz_neg_x2_std_k1_1_ref=np.std(first_values_yz_neg_x2_list_k1_1_ref, axis=0)
    first_values_xz_pos_y2_std_k1_1_ref=np.std(first_values_xz_pos_y2_list_k1_1_ref, axis=0)
    first_values_xz_neg_y2_std_k1_1_ref=np.std(first_values_xz_neg_y2_list_k1_1_ref, axis=0)
    
    region_mean_values_ref_1=np.array(region_mean_values_ref_1)
    region_mean_k1_1_ref=np.mean(region_mean_values_ref_1, axis=0)
    region_std_k1_1_ref=np.std(region_mean_values_ref_1, axis=0)
    
    
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(regions, flow_mean_k1_1, flow_std_k1_1)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "flow_region_mean_std_k1_1.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
        
    mean_and_std_first_values_yz_neg_x0_k1_1=[first_values_yz_neg_x0_mean_k1_1, first_values_yz_neg_x0_std_k1_1]
    mean_and_std_first_values_yz_pos_x1_k1_1=[first_values_yz_pos_x1_mean_k1_1, first_values_yz_pos_x1_std_k1_1]
    mean_and_std_first_values_yz_pos_x2_k1_1=[first_values_yz_pos_x2_mean_k1_1, first_values_yz_pos_x2_std_k1_1]
    mean_and_std_first_values_yz_neg_x2_k1_1=[first_values_yz_neg_x2_mean_k1_1, first_values_yz_neg_x2_std_k1_1]
    mean_and_std_first_values_xz_pos_y2_k1_1=[first_values_xz_pos_y2_mean_k1_1, first_values_xz_pos_y2_std_k1_1]
    mean_and_std_first_values_xz_neg_y2_k1_1=[first_values_xz_neg_y2_mean_k1_1, first_values_xz_neg_y2_std_k1_1]
    
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x0_k1_1.npy'), mean_and_std_first_values_yz_neg_x0_k1_1)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x1_k1_1.npy'), mean_and_std_first_values_yz_pos_x1_k1_1)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x2_k1_1.npy'), mean_and_std_first_values_yz_pos_x2_k1_1)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x2_k1_1.npy'), mean_and_std_first_values_yz_neg_x2_k1_1)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_pos_y2_k1_1.npy'), mean_and_std_first_values_xz_pos_y2_k1_1)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_neg_y2_k1_1.npy'), mean_and_std_first_values_xz_neg_y2_k1_1)
    
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(mask_names, region_mean_k1_1, region_std_k1_1)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "48_region_mean_std_k1_1.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
    
    #reference:
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(regions, flow_mean_k1_1_ref, flow_std_k1_1_ref)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "flow_region_mean_std_k1_1_ref.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
        
    mean_and_std_first_values_yz_neg_x0_k1_1_ref=[first_values_yz_neg_x0_mean_k1_1_ref, first_values_yz_neg_x0_std_k1_1_ref]
    mean_and_std_first_values_yz_pos_x1_k1_1_ref=[first_values_yz_pos_x1_mean_k1_1_ref, first_values_yz_pos_x1_std_k1_1_ref]
    mean_and_std_first_values_yz_pos_x2_k1_1_ref=[first_values_yz_pos_x2_mean_k1_1_ref, first_values_yz_pos_x2_std_k1_1_ref]
    mean_and_std_first_values_yz_neg_x2_k1_1_ref=[first_values_yz_neg_x2_mean_k1_1_ref, first_values_yz_neg_x2_std_k1_1_ref]
    mean_and_std_first_values_xz_pos_y2_k1_1_ref=[first_values_xz_pos_y2_mean_k1_1_ref, first_values_xz_pos_y2_std_k1_1_ref]
    mean_and_std_first_values_xz_neg_y2_k1_1_ref=[first_values_xz_neg_y2_mean_k1_1_ref, first_values_xz_neg_y2_std_k1_1_ref]
    
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x0_k1_1_ref.npy'), mean_and_std_first_values_yz_neg_x0_k1_1_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x1_k1_1_ref.npy'), mean_and_std_first_values_yz_pos_x1_k1_1_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x2_k1_1_ref.npy'), mean_and_std_first_values_yz_pos_x2_k1_1_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x2_k1_1_ref.npy'), mean_and_std_first_values_yz_neg_x2_k1_1_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_pos_y2_k1_1_ref.npy'), mean_and_std_first_values_xz_pos_y2_k1_1_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_neg_y2_k1_1_ref.npy'), mean_and_std_first_values_xz_neg_y2_k1_1_ref)
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(mask_names, region_mean_k1_1_ref, region_std_k1_1_ref)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "48_region_mean_std_k1_1_ref.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
        
    
    # zip_path = r"C:\Users\jacke\OneDrive\Skrivbord\Wat2.zip"
    data_array_2 = load_nifti_data(path_2)
    # print('data_array', data_array_2.shape)
    
    data_array_2=np.rot90(data_array_2, k=2, axes=(1,2))
    data_array_2=np.rot90(data_array_2, k=2, axes=(0,1))
    
    flow_mean_list_all_2=[]
    
    first_values_yz_neg_x0_list_k1_2=[]
    first_values_yz_pos_x1_list_k1_2=[]
    first_values_yz_pos_x2_list_k1_2=[]
    first_values_yz_neg_x2_list_k1_2=[]
    first_values_xz_pos_y2_list_k1_2=[]
    first_values_xz_neg_y2_list_k1_2=[]
    
    region_mean_values_2=[]
    
    flow_mean_list_all_ref_2=[]
    
    first_values_yz_neg_x0_list_k1_2_ref=[]
    first_values_yz_pos_x1_list_k1_2_ref=[]
    first_values_yz_pos_x2_list_k1_2_ref=[]
    first_values_yz_neg_x2_list_k1_2_ref=[]
    first_values_xz_pos_y2_list_k1_2_ref=[]
    first_values_xz_neg_y2_list_k1_2_ref=[]
    
    region_mean_values_ref_2=[]
    
    for i in range(np.shape(data_array_2)[3]):
           
        from SSP_2d import SSP_2D
        first_values_yz_neg_x0, first_values_yz_pos_x1, first_values_yz_pos_x2, first_values_yz_neg_x2, first_values_xz_pos_y2, first_values_xz_neg_y2=SSP_2D(data_array_2[..., i])
        first_values_yz_neg_x0_list_k1_2.append(first_values_yz_neg_x0)
        first_values_yz_pos_x1_list_k1_2.append(first_values_yz_pos_x1)
        first_values_yz_pos_x2_list_k1_2.append(first_values_yz_pos_x2)
        first_values_yz_neg_x2_list_k1_2.append(first_values_yz_neg_x2)
        first_values_xz_pos_y2_list_k1_2.append(first_values_xz_pos_y2)
        first_values_xz_neg_y2_list_k1_2.append(first_values_xz_neg_y2)
        
        from Z_score import SD_corrected
        ålder=1 
        kön='M'
        frame='real'
        wat='wat1'
        igon, igon, Cerebellum_mean_k1, flow_mean_list =SD_corrected(data_array_2[..., i], ålder, kön, frame, wat)
        flow_mean_list_all_2.append(flow_mean_list)
        # # print(flow_mean_list)
        
        from plot_regioner_mean import regions_z_score
        
        igno, igno, igno, igno, igno, igno, igno, igno, mean_values=regions_z_score(data_array_2[..., i], frame, wat)
        
        #mask_names is the key to the mean values    
        region_mean_values_temp=[]
        # print('i=', i)
        for name in mask_names:
            region_mean_values_temp.append(mean_values[name])
            # # print(f'{name}: mean= {mean_values[name]}')
        region_mean_values_2.append(region_mean_values_temp)
        
        igon, igno, Cerebellum_mean_k1_ref, flow_mean_list_ref =SD_corrected(data_array_2[..., i]/Cerebellum_mean_k1, ålder, kön, frame, wat)
        flow_mean_list_all_ref_2.append(flow_mean_list_ref)
        
        
        #reference
        
        igno, igno, igno, igno, igno, igno, igon, igon, mean_values_ref = regions_z_score(data_array_2[..., i]/Cerebellum_mean_k1, 'ref', 'wat1')
        
        #mask_names is the key to the mean values    
        region_mean_values_temp_ref=[]
        # print('i=', i)
        for name in mask_names:
            region_mean_values_temp_ref.append(mean_values_ref[name])
            # # print(f'{name}: mean= {mean_values[name]}')
        region_mean_values_ref_2.append(region_mean_values_temp_ref)
    
        
        first_values_yz_neg_x0_ref, first_values_yz_pos_x1_ref, first_values_yz_pos_x2_ref, first_values_yz_neg_x2_ref, first_values_xz_pos_y2_ref, first_values_xz_neg_y2_ref=first_values_yz_neg_x0/Cerebellum_mean_k1, first_values_yz_pos_x1/Cerebellum_mean_k1, first_values_yz_pos_x2/Cerebellum_mean_k1, first_values_yz_neg_x2/Cerebellum_mean_k1, first_values_xz_pos_y2/Cerebellum_mean_k1, first_values_xz_neg_y2/Cerebellum_mean_k1
        first_values_yz_neg_x0_list_k1_2_ref.append(first_values_yz_neg_x0_ref)
        first_values_yz_pos_x1_list_k1_2_ref.append(first_values_yz_pos_x1_ref)
        first_values_yz_pos_x2_list_k1_2_ref.append(first_values_yz_pos_x2_ref)
        first_values_yz_neg_x2_list_k1_2_ref.append(first_values_yz_neg_x2_ref)
        first_values_xz_pos_y2_list_k1_2_ref.append(first_values_xz_pos_y2_ref)
        first_values_xz_neg_y2_list_k1_2_ref.append(first_values_xz_neg_y2_ref)
        
    
        
    # print('total tid:', time.time()-start_tid)
    
    flow_mean_list_all_2=np.array(flow_mean_list_all_2)
    flow_mean_k1_2=np.mean(flow_mean_list_all_2, axis=0)
    flow_std_k1_2=np.std(flow_mean_list_all_2, axis=0)
    
    
    first_values_yz_neg_x0_mean_k1_2=np.mean(first_values_yz_neg_x0_list_k1_2, axis=0)
    first_values_yz_pos_x1_mean_k1_2=np.mean(first_values_yz_pos_x1_list_k1_2, axis=0)
    first_values_yz_pos_x2_mean_k1_2=np.mean(first_values_yz_pos_x2_list_k1_2, axis=0)
    first_values_yz_neg_x2_mean_k1_2=np.mean(first_values_yz_neg_x2_list_k1_2, axis=0)
    first_values_xz_pos_y2_mean_k1_2=np.mean(first_values_xz_pos_y2_list_k1_2, axis=0)
    first_values_xz_neg_y2_mean_k1_2=np.mean(first_values_xz_neg_y2_list_k1_2, axis=0)
    
    first_values_yz_neg_x0_std_k1_2=np.std(first_values_yz_neg_x0_list_k1_2, axis=0)
    first_values_yz_pos_x1_std_k1_2=np.std(first_values_yz_pos_x1_list_k1_2, axis=0)
    first_values_yz_pos_x2_std_k1_2=np.std(first_values_yz_pos_x2_list_k1_2, axis=0)
    first_values_yz_neg_x2_std_k1_2=np.std(first_values_yz_neg_x2_list_k1_2, axis=0)
    first_values_xz_pos_y2_std_k1_2=np.std(first_values_xz_pos_y2_list_k1_2, axis=0)
    first_values_xz_neg_y2_std_k1_2=np.std(first_values_xz_neg_y2_list_k1_2, axis=0)
    
    
    region_mean_values_2=np.array(region_mean_values_2)
    region_mean_k1_2=np.mean(region_mean_values_2, axis=0)
    region_std_k1_2=np.std(region_mean_values_2, axis=0)
    
    
    
    
    flow_mean_list_all_ref_2=np.array(flow_mean_list_all_ref_2)
    flow_mean_k1_2_ref=np.mean(flow_mean_list_all_ref_2, axis=0)
    flow_std_k1_2_ref=np.std(flow_mean_list_all_ref_2, axis=0)
    
    first_values_yz_neg_x0_mean_k1_2_ref=np.mean(first_values_yz_neg_x0_list_k1_2_ref, axis=0)
    first_values_yz_pos_x1_mean_k1_2_ref=np.mean(first_values_yz_pos_x1_list_k1_2_ref, axis=0)
    first_values_yz_pos_x2_mean_k1_2_ref=np.mean(first_values_yz_pos_x2_list_k1_2_ref, axis=0)
    first_values_yz_neg_x2_mean_k1_2_ref=np.mean(first_values_yz_neg_x2_list_k1_2_ref, axis=0)
    first_values_xz_pos_y2_mean_k1_2_ref=np.mean(first_values_xz_pos_y2_list_k1_2_ref, axis=0)
    first_values_xz_neg_y2_mean_k1_2_ref=np.mean(first_values_xz_neg_y2_list_k1_2_ref, axis=0)
    
    first_values_yz_neg_x0_std_k1_2_ref=np.std(first_values_yz_neg_x0_list_k1_2_ref, axis=0)
    first_values_yz_pos_x1_std_k1_2_ref=np.std(first_values_yz_pos_x1_list_k1_2_ref, axis=0)
    first_values_yz_pos_x2_std_k1_2_ref=np.std(first_values_yz_pos_x2_list_k1_2_ref, axis=0)
    first_values_yz_neg_x2_std_k1_2_ref=np.std(first_values_yz_neg_x2_list_k1_2_ref, axis=0)
    first_values_xz_pos_y2_std_k1_2_ref=np.std(first_values_xz_pos_y2_list_k1_2_ref, axis=0)
    first_values_xz_neg_y2_std_k1_2_ref=np.std(first_values_xz_neg_y2_list_k1_2_ref, axis=0)
    
    region_mean_values_ref_2=np.array(region_mean_values_ref_2)
    region_mean_k1_2_ref=np.mean(region_mean_values_ref_2, axis=0)
    region_std_k1_2_ref=np.std(region_mean_values_ref_2, axis=0)
    
    
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(regions, flow_mean_k1_2, flow_std_k1_2)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "flow_region_mean_std_k1_2.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
        
    mean_and_std_first_values_yz_neg_x0_k1_2=[first_values_yz_neg_x0_mean_k1_2, first_values_yz_neg_x0_std_k1_2]
    mean_and_std_first_values_yz_pos_x1_k1_2=[first_values_yz_pos_x1_mean_k1_2, first_values_yz_pos_x1_std_k1_2]
    mean_and_std_first_values_yz_pos_x2_k1_2=[first_values_yz_pos_x2_mean_k1_2, first_values_yz_pos_x2_std_k1_2]
    mean_and_std_first_values_yz_neg_x2_k1_2=[first_values_yz_neg_x2_mean_k1_2, first_values_yz_neg_x2_std_k1_2]
    mean_and_std_first_values_xz_pos_y2_k1_2=[first_values_xz_pos_y2_mean_k1_2, first_values_xz_pos_y2_std_k1_2]
    mean_and_std_first_values_xz_neg_y2_k1_2=[first_values_xz_neg_y2_mean_k1_2, first_values_xz_neg_y2_std_k1_2]
    
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x0_k1_2.npy'), mean_and_std_first_values_yz_neg_x0_k1_2)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x1_k1_2.npy'), mean_and_std_first_values_yz_pos_x1_k1_2)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x2_k1_2.npy'), mean_and_std_first_values_yz_pos_x2_k1_2)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x2_k1_2.npy'), mean_and_std_first_values_yz_neg_x2_k1_2)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_pos_y2_k1_2.npy'), mean_and_std_first_values_xz_pos_y2_k1_2)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_neg_y2_k1_2.npy'), mean_and_std_first_values_xz_neg_y2_k1_2)
    
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(mask_names, region_mean_k1_2, region_std_k1_2)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "48_region_mean_std_k1_2.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
    
    #reference:
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(regions, flow_mean_k1_2_ref, flow_std_k1_2_ref)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "flow_region_mean_std_k1_2_ref.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
        
    mean_and_std_first_values_yz_neg_x0_k1_2_ref=[first_values_yz_neg_x0_mean_k1_2_ref, first_values_yz_neg_x0_std_k1_2_ref]
    mean_and_std_first_values_yz_pos_x1_k1_2_ref=[first_values_yz_pos_x1_mean_k1_2_ref, first_values_yz_pos_x1_std_k1_2_ref]
    mean_and_std_first_values_yz_pos_x2_k1_2_ref=[first_values_yz_pos_x2_mean_k1_2_ref, first_values_yz_pos_x2_std_k1_2_ref]
    mean_and_std_first_values_yz_neg_x2_k1_2_ref=[first_values_yz_neg_x2_mean_k1_2_ref, first_values_yz_neg_x2_std_k1_2_ref]
    mean_and_std_first_values_xz_pos_y2_k1_2_ref=[first_values_xz_pos_y2_mean_k1_2_ref, first_values_xz_pos_y2_std_k1_2_ref]
    mean_and_std_first_values_xz_neg_y2_k1_2_ref=[first_values_xz_neg_y2_mean_k1_2_ref, first_values_xz_neg_y2_std_k1_2_ref]
    
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x0_k1_2_ref.npy'), mean_and_std_first_values_yz_neg_x0_k1_2_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x1_k1_2_ref.npy'), mean_and_std_first_values_yz_pos_x1_k1_2_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x2_k1_2_ref.npy'), mean_and_std_first_values_yz_pos_x2_k1_2_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x2_k1_2_ref.npy'), mean_and_std_first_values_yz_neg_x2_k1_2_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_pos_y2_k1_2_ref.npy'), mean_and_std_first_values_xz_pos_y2_k1_2_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_neg_y2_k1_2_ref.npy'), mean_and_std_first_values_xz_neg_y2_k1_2_ref)
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(mask_names, region_mean_k1_2_ref, region_std_k1_2_ref)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "48_region_mean_std_k1_2_ref.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
    #%%
    data_array_3=data_array_2[...,:2]/data_array_1[..., :2]
    
    flow_mean_list_all_3=[]
    
    first_values_yz_neg_x0_list_k1_3=[]
    first_values_yz_pos_x1_list_k1_3=[]
    first_values_yz_pos_x2_list_k1_3=[]
    first_values_yz_neg_x2_list_k1_3=[]
    first_values_xz_pos_y2_list_k1_3=[]
    first_values_xz_neg_y2_list_k1_3=[]
    
    region_mean_values_3=[]
    
    flow_mean_list_all_ref_3=[]
    
    first_values_yz_neg_x0_list_k1_3_ref=[]
    first_values_yz_pos_x1_list_k1_3_ref=[]
    first_values_yz_pos_x2_list_k1_3_ref=[]
    first_values_yz_neg_x2_list_k1_3_ref=[]
    first_values_xz_pos_y2_list_k1_3_ref=[]
    first_values_xz_neg_y2_list_k1_3_ref=[]
    
    region_mean_values_ref_3=[]
    
    for i in range(np.shape(data_array_3)[3]):
           
        
        from Z_score import SD_corrected
        ålder=1 
        kön='M'
        frame='real'
        wat='wat1'
        igon, igon, Cerebellum_mean_k1, flow_mean_list =SD_corrected(data_array_3[..., i], ålder, kön, frame, wat)
        flow_mean_list_all_3.append(flow_mean_list)
        # # print(flow_mean_list)
        
        from plot_regioner_mean import regions_z_score
        
        igno, igno, igno, igno, igno, igno, igno, igno, mean_values=regions_z_score(data_array_3[..., i], frame, wat)
        
        #mask_names is the key to the mean values    
        region_mean_values_temp=[]
        # print('i=', i)
        for name in mask_names:
            region_mean_values_temp.append(mean_values[name])
            # # print(f'{name}: mean= {mean_values[name]}')
        region_mean_values_3.append(region_mean_values_temp)
        
        igon, igno, Cerebellum_mean_k1_ref, flow_mean_list_ref =SD_corrected(data_array_3[..., i]/Cerebellum_mean_k1, ålder, kön, frame, wat)
        flow_mean_list_all_ref_3.append(flow_mean_list_ref)
        
        
        #reference
        
        igno, igno, igno, igno, igno, igno, igon, igon, mean_values_ref = regions_z_score(data_array_3[..., i]/Cerebellum_mean_k1, 'ref', 'wat1')
        
        #mask_names is the key to the mean values    
        region_mean_values_temp_ref=[]
        # print('i=', i)
        for name in mask_names:
            region_mean_values_temp_ref.append(mean_values_ref[name])
            # # print(f'{name}: mean= {mean_values[name]}')
        region_mean_values_ref_3.append(region_mean_values_temp_ref)
        
    
        
    # print('total tid:', time.time()-start_tid)
    #%%
    
    flow_mean_list_all_3=np.array(flow_mean_list_all_3)
    flow_mean_k1_3=np.mean(flow_mean_list_all_3, axis=0)
    flow_std_k1_3=np.std(flow_mean_list_all_3, axis=0)
    #%%
    
    first_values_yz_neg_x0_mean_k1_3=np.mean(np.array(first_values_yz_neg_x0_list_k1_2)/np.array(first_values_yz_neg_x0_list_k1_1), axis=0)
    first_values_yz_pos_x1_mean_k1_3=np.mean(np.array(first_values_yz_pos_x1_list_k1_2)/np.array(first_values_yz_pos_x1_list_k1_1), axis=0)
    first_values_yz_pos_x2_mean_k1_3=np.mean(np.array(first_values_yz_pos_x2_list_k1_2)/np.array(first_values_yz_pos_x2_list_k1_1), axis=0)
    first_values_yz_neg_x2_mean_k1_3=np.mean(np.array(first_values_yz_neg_x2_list_k1_2)/np.array(first_values_yz_neg_x2_list_k1_1), axis=0)
    first_values_xz_pos_y2_mean_k1_3=np.mean(np.array(first_values_xz_pos_y2_list_k1_2)/np.array(first_values_xz_pos_y2_list_k1_1), axis=0)
    first_values_xz_neg_y2_mean_k1_3=np.mean(np.array(first_values_xz_neg_y2_list_k1_2)/np.array(first_values_xz_neg_y2_list_k1_1), axis=0)
    
    first_values_yz_neg_x0_std_k1_3=np.std(np.array(first_values_yz_neg_x0_list_k1_2)/np.array(first_values_yz_neg_x0_list_k1_1), axis=0)
    first_values_yz_pos_x1_std_k1_3=np.std(np.array(first_values_yz_pos_x1_list_k1_2)/np.array(first_values_yz_pos_x1_list_k1_1), axis=0)
    first_values_yz_pos_x2_std_k1_3=np.std(np.array(first_values_yz_pos_x2_list_k1_2)/np.array(first_values_yz_pos_x2_list_k1_1), axis=0)
    first_values_yz_neg_x2_std_k1_3=np.std(np.array(first_values_yz_neg_x2_list_k1_2)/np.array(first_values_yz_neg_x2_list_k1_1), axis=0)
    first_values_xz_pos_y2_std_k1_3=np.std(np.array(first_values_xz_pos_y2_list_k1_2)/np.array(first_values_xz_pos_y2_list_k1_1), axis=0)
    first_values_xz_neg_y2_std_k1_3=np.std(np.array(first_values_xz_neg_y2_list_k1_2)/np.array(first_values_xz_neg_y2_list_k1_1), axis=0)
    
    #%%
    region_mean_values_3=np.array(region_mean_values_3)
    region_mean_k1_3=np.mean(region_mean_values_3, axis=0)
    region_std_k1_3=np.std(region_mean_values_3, axis=0)
    
    
    
    
    flow_mean_list_all_ref_3=np.array(flow_mean_list_all_ref_3)
    flow_mean_k1_3_ref=np.mean(flow_mean_list_all_ref_3, axis=0)
    flow_std_k1_3_ref=np.std(flow_mean_list_all_ref_3, axis=0)
    
    first_values_yz_neg_x0_mean_k1_3_ref=np.mean(np.array(first_values_yz_neg_x0_list_k1_2_ref)/np.array(first_values_yz_neg_x0_list_k1_1_ref), axis=0)
    first_values_yz_pos_x1_mean_k1_3_ref=np.mean(np.array(first_values_yz_pos_x1_list_k1_2_ref)/np.array(first_values_yz_pos_x1_list_k1_1_ref), axis=0)
    first_values_yz_pos_x2_mean_k1_3_ref=np.mean(np.array(first_values_yz_pos_x2_list_k1_2_ref)/np.array(first_values_yz_pos_x2_list_k1_1_ref), axis=0)
    first_values_yz_neg_x2_mean_k1_3_ref=np.mean(np.array(first_values_yz_neg_x2_list_k1_2_ref)/np.array(first_values_yz_neg_x2_list_k1_1_ref), axis=0)
    first_values_xz_pos_y2_mean_k1_3_ref=np.mean(np.array(first_values_xz_pos_y2_list_k1_2_ref)/np.array(first_values_xz_pos_y2_list_k1_1_ref), axis=0)
    first_values_xz_neg_y2_mean_k1_3_ref=np.mean(np.array(first_values_xz_neg_y2_list_k1_2_ref)/np.array(first_values_xz_neg_y2_list_k1_1_ref), axis=0)
    
    first_values_yz_neg_x0_std_k1_3_ref=np.std(np.array(first_values_yz_neg_x0_list_k1_2_ref)/np.array(first_values_yz_neg_x0_list_k1_1_ref), axis=0)
    first_values_yz_pos_x1_std_k1_3_ref=np.std(np.array(first_values_yz_pos_x1_list_k1_2_ref)/np.array(first_values_yz_pos_x1_list_k1_1_ref), axis=0)
    first_values_yz_pos_x2_std_k1_3_ref=np.std(np.array(first_values_yz_pos_x2_list_k1_2_ref)/np.array(first_values_yz_pos_x2_list_k1_1_ref), axis=0)
    first_values_yz_neg_x2_std_k1_3_ref=np.std(np.array(first_values_yz_neg_x2_list_k1_2_ref)/np.array(first_values_yz_neg_x2_list_k1_1_ref), axis=0)
    first_values_xz_pos_y2_std_k1_3_ref=np.std(np.array(first_values_xz_pos_y2_list_k1_2_ref)/np.array(first_values_xz_pos_y2_list_k1_1_ref), axis=0)
    first_values_xz_neg_y2_std_k1_3_ref=np.std(np.array(first_values_xz_neg_y2_list_k1_2_ref)/np.array(first_values_xz_neg_y2_list_k1_1_ref), axis=0)
    
    
    region_mean_values_ref_3=np.array(region_mean_values_ref_3)
    region_mean_k1_3_ref=np.mean(region_mean_values_ref_3, axis=0)
    region_std_k1_3_ref=np.std(region_mean_values_ref_3, axis=0)
    
    
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(regions, flow_mean_k1_3, flow_std_k1_3)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "flow_region_mean_std_k1_3.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
        
    mean_and_std_first_values_yz_neg_x0_k1_3=[first_values_yz_neg_x0_mean_k1_3, first_values_yz_neg_x0_std_k1_3]
    mean_and_std_first_values_yz_pos_x1_k1_3=[first_values_yz_pos_x1_mean_k1_3, first_values_yz_pos_x1_std_k1_3]
    mean_and_std_first_values_yz_pos_x2_k1_3=[first_values_yz_pos_x2_mean_k1_3, first_values_yz_pos_x2_std_k1_3]
    mean_and_std_first_values_yz_neg_x2_k1_3=[first_values_yz_neg_x2_mean_k1_3, first_values_yz_neg_x2_std_k1_3]
    mean_and_std_first_values_xz_pos_y2_k1_3=[first_values_xz_pos_y2_mean_k1_3, first_values_xz_pos_y2_std_k1_3]
    mean_and_std_first_values_xz_neg_y2_k1_3=[first_values_xz_neg_y2_mean_k1_3, first_values_xz_neg_y2_std_k1_3]
    
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x0_k1_3.npy'), mean_and_std_first_values_yz_neg_x0_k1_3)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x1_k1_3.npy'), mean_and_std_first_values_yz_pos_x1_k1_3)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x2_k1_3.npy'), mean_and_std_first_values_yz_pos_x2_k1_3)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x2_k1_3.npy'), mean_and_std_first_values_yz_neg_x2_k1_3)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_pos_y2_k1_3.npy'), mean_and_std_first_values_xz_pos_y2_k1_3)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_neg_y2_k1_3.npy'), mean_and_std_first_values_xz_neg_y2_k1_3)
    
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(mask_names, region_mean_k1_3, region_std_k1_3)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "48_region_mean_std_k1_3.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
    
    #reference:
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(regions, flow_mean_k1_3_ref, flow_std_k1_3_ref)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "flow_region_mean_std_k1_3_ref.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
        
    mean_and_std_first_values_yz_neg_x0_k1_3_ref=[first_values_yz_neg_x0_mean_k1_3_ref, first_values_yz_neg_x0_std_k1_3_ref]
    mean_and_std_first_values_yz_pos_x1_k1_3_ref=[first_values_yz_pos_x1_mean_k1_3_ref, first_values_yz_pos_x1_std_k1_3_ref]
    mean_and_std_first_values_yz_pos_x2_k1_3_ref=[first_values_yz_pos_x2_mean_k1_3_ref, first_values_yz_pos_x2_std_k1_3_ref]
    mean_and_std_first_values_yz_neg_x2_k1_3_ref=[first_values_yz_neg_x2_mean_k1_3_ref, first_values_yz_neg_x2_std_k1_3_ref]
    mean_and_std_first_values_xz_pos_y2_k1_3_ref=[first_values_xz_pos_y2_mean_k1_3_ref, first_values_xz_pos_y2_std_k1_3_ref]
    mean_and_std_first_values_xz_neg_y2_k1_3_ref=[first_values_xz_neg_y2_mean_k1_3_ref, first_values_xz_neg_y2_std_k1_3_ref]
    
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x0_k1_3_ref.npy'), mean_and_std_first_values_yz_neg_x0_k1_3_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x1_k1_3_ref.npy'), mean_and_std_first_values_yz_pos_x1_k1_3_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_pos_x2_k1_3_ref.npy'), mean_and_std_first_values_yz_pos_x2_k1_3_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_yz_neg_x2_k1_3_ref.npy'), mean_and_std_first_values_yz_neg_x2_k1_3_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_pos_y2_k1_3_ref.npy'), mean_and_std_first_values_xz_pos_y2_k1_3_ref)
    np.save(os.path.join(out_path, 'mean_and_std_first_values_xz_neg_y2_k1_3_ref.npy'), mean_and_std_first_values_xz_neg_y2_k1_3_ref)
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(mask_names, region_mean_k1_3_ref, region_std_k1_3_ref)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "48_region_mean_std_k1_3_ref.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)

# path_1 = r"C:\Users\jacke\OneDrive\Skrivbord\Wat1.zip"
# path_2 = r"C:\Users\jacke\OneDrive\Skrivbord\Wat2.zip"
# wat1_wat2_mean_std(path_1, path_2, out_path)

def R_I_mean_std(path, out_path):
    # print('kör R_I med', path)
    data_array = load_nifti_data(path)
    
    data_array=np.rot90(data_array, k=2, axes=(1,2))
    data_array=np.rot90(data_array, k=2, axes=(0,1))
    
    flow_mean_list_all=[]
    
    region_mean_values=[]
    
    flow_mean_list_all_ref=[]
    
    
    region_mean_values_ref=[]
    
    for i in range(np.shape(data_array)[3]):
    
        
        from Z_score import SD_corrected_park
        ålder=1 
        kön='M'
        frame='real'
        wat='wat1'
        Z_brain, z_scores_flow, means_list =SD_corrected_park(data_array[...,i], tracer='C-11')
    
        flow_mean_list_all.append(means_list)
        # # print(flow_mean_list)
        
        from plot_regioner_mean_parkinson import regions_z_score_park
        
        igno, igno, igno, igno, igno, igno, igno, igno, mean_values=regions_z_score_park(data_array[..., i], tracer='C-11')
        
        #mask_names is the key to the mean values    
        region_mean_values_temp=[]
        # print('i=', i)
        for name in mask_names:
            region_mean_values_temp.append(mean_values[name])
            # # print(f'{name}: mean= {mean_values[name]}')
        region_mean_values.append(region_mean_values_temp)
    
    # print('total tid:', time.time()-start_tid)
    
    flow_mean_list_all=np.array(flow_mean_list_all)
    flow_mean_R_I=np.mean(flow_mean_list_all, axis=0)
    flow_std_R_I=np.std(flow_mean_list_all, axis=0)
    
    
    region_mean_values=np.array(region_mean_values)
    region_mean_R_I=np.mean(region_mean_values, axis=0)
    region_std_R_I=np.std(region_mean_values, axis=0)
    
    
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(regions, flow_mean_R_I, flow_std_R_I)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "flow_region_mean_std_R_I.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
    
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(mask_names, region_mean_R_I, region_std_R_I)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "48_region_mean_std_R_I.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
# path = r"C:\Users\jacke\OneDrive\Skrivbord\R_I_nii.zip"
# R_I_mean_std(path, out_path)

def R_I_C_mean_std(path, out_path):
    # print('kör R_I_C med', path)
    data_array = load_nifti_data(path)
    
    data_array=np.rot90(data_array, k=2, axes=(1,2))
    data_array=np.rot90(data_array, k=2, axes=(0,1))
    
    flow_mean_list_all=[]
    
    region_mean_values=[]
    
    flow_mean_list_all_ref=[]
    
    
    region_mean_values_ref=[]
    
    for i in range(np.shape(data_array)[3]):
    
        
        from Z_score import SD_corrected_park
        Z_brain, z_scores_flow, means_list =SD_corrected_park(data_array[...,i], tracer='C-11')
    
        flow_mean_list_all.append(means_list)
        # # print(flow_mean_list)
        
        from plot_regioner_mean_parkinson import regions_z_score_park
        
        igno, igno, igno, igno, igno, igno, igno, igno, mean_values=regions_z_score_park(data_array[..., i], tracer='C-11')
        
        #mask_names is the key to the mean values    
        region_mean_values_temp=[]
        # print('i=', i)
        for name in mask_names:
            region_mean_values_temp.append(mean_values[name])
            # # print(f'{name}: mean= {mean_values[name]}')
        region_mean_values.append(region_mean_values_temp)
    
    # print('total tid:', time.time()-start_tid)
    
    flow_mean_list_all=np.array(flow_mean_list_all)
    flow_mean_R_I_C=np.mean(flow_mean_list_all, axis=0)
    flow_std_R_I_C=np.std(flow_mean_list_all, axis=0)
    
    
    region_mean_values=np.array(region_mean_values)
    region_mean_R_I_C=np.mean(region_mean_values, axis=0)
    region_std_R_I_C=np.std(region_mean_values, axis=0)
    
    
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(regions, flow_mean_R_I_C, flow_std_R_I_C)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path,  "flow_region_mean_std_R_I_C.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)
    
        
    header = "%Region, Mean, Std\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    data_lines = [f"{region}, {mean:.3f}, {std:.3f}" for region, mean, std in zip(mask_names, region_mean_R_I_C, region_std_R_I_C)]
    
    # Write to a .txt file
    output_content = header + "\n".join(data_lines)
    output_file_path =os.path.join(out_path, "48_region_mean_std_R_I_C.txt")
    with open(output_file_path, "w") as file:
        file.write(output_content)

# path = r"C:\Users\jacke\OneDrive\Skrivbord\R_I_C_nii.zip"
# R_I_C_mean_std(path, out_path)


def backup_existing_files(out_path, filenames):
    """
    For each filename in filenames, check if the file exists in out_path.
    If it does, create a backup copy with '-backup' appended before the extension.
    """
    for filename in filenames:
        full_path = os.path.join(out_path, filename)
        if os.path.isfile(full_path):
            # Create backup filename
            name, ext = os.path.splitext(filename)
            backup_filename = f"{name}-backup{ext}"
            backup_full_path = os.path.join(out_path, backup_filename)
            # Copy the file
            shutil.copy2(full_path, backup_full_path)
            # print(f"Created backup of {filename} as {backup_filename}")



# -*- coding: utf-8 -*-
"""
Tkinter GUI Application for Flow and SSP Calculations

Features:
- Main menu with four processing options: k1, wat2/wat1, F-18, C-11.
- Each processing option allows selecting input via folder or ZIP file.
- Only the non-selected input button is deactivated with a distinct color.
- "Run" button has a green outline for emphasis.
- Returns to the main menu after successful processing.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Import your processing functions
# Make sure these functions are properly defined in your codebase or imported from the appropriate modules
# from your_module import water_mean_std, wat1_wat2_mean_std, R_I_mean_std, R_I_C_mean_std

# -------------------------------
# Tkinter GUI Implementation
# -------------------------------

def main():
    root = tk.Tk()
    root.title("Flow and SSP Calculations")
    root.geometry("800x500")
    root.configure(bg="black")

    style = ttk.Style()
    style.theme_use("clam")

    # Configure the general TButton style
    style.configure("TButton",
                    background="grey",
                    foreground="white",
                    font=("Arial", 10),
                    borderwidth=1,
                    relief="solid")
    style.map("TButton",
              background=[('active', 'darkgrey'), ('disabled', 'dim gray')],
              foreground=[('active', 'white'), ('disabled', 'light gray')])

    # Configure the Run.TButton style (additional styling for the Run button)
    style.configure("Run.TButton",
                    background="grey",
                    foreground="white",
                    font=("Arial", 10),
                    borderwidth=0,
                    relief="flat")
    style.map("Run.TButton",
              background=[('active', 'darkgrey')],
              foreground=[('active', 'white')])

    # Create a main frame using tk.Frame to allow background configuration
    main_frame = tk.Frame(root, bg="black", padx=10, pady=10)
    main_frame.pack(expand=True, fill='both')

    # Add label using tk.Label
    label = tk.Label(main_frame, text="Välj vilken du vill skapa nytt normalmaterial för",
                     bg="black", fg="white", font=("Arial", 14))
    label.pack(pady=20)

    # Create a frame for the buttons using tk.Frame
    button_frame = tk.Frame(main_frame, bg="black")
    button_frame.pack(pady=10)

    # Create four buttons: k1, wat2/wat1, F-18, C-11 using ttk.Button
    button_k1 = ttk.Button(button_frame, text="O-15", command=lambda: process_k1(root))
    button_wat = ttk.Button(button_frame, text="O-15 (stress/basline)", command=lambda: process_wat(root))
    button_F18 = ttk.Button(button_frame, text="F-18", command=lambda: process_F18(root))
    button_C11 = ttk.Button(button_frame, text="C-11", command=lambda: process_C11(root))

    # Place the buttons in the same row with some padding
    button_k1.grid(row=0, column=0, padx=10, pady=10)
    button_wat.grid(row=0, column=1, padx=10, pady=10)
    button_F18.grid(row=0, column=2, padx=10, pady=10)
    button_C11.grid(row=0, column=3, padx=10, pady=10)

    root.mainloop()


def process_k1(root):
    """
    Processing window for k1.
    Allows selecting one input (folder or ZIP) and an output folder.
    """
    clear_window(root)

    # Create a main frame
    main_frame = tk.Frame(root, bg="black", padx=10, pady=10)
    main_frame.pack(expand=True, fill='both')

    # Add label
    label = tk.Label(main_frame, text="O-15 Vatten", bg="black", fg="white", font=("Arial", 14))
    label.pack(pady=10)

    # Input information
    input_info = {'path': None, 'type': None}
    output_path = tk.StringVar()

    # Define button toggle function
    def toggle_buttons(active_button, inactive_button):
        inactive_button.config(state="disabled")
        # Update styles if needed
        # active_button remains enabled

    def select_input_directory():
        path = filedialog.askdirectory()
        if path:
            input_info['path'] = path
            input_info['type'] = 'Directory'
            button_dir_input.config(text=path)
            button_zip_input.config(text="Välj zip", state='disabled', style="TButton")
            toggle_buttons(button_dir_input, button_zip_input)
            # print(f"Selected input directory: {path}")

    def select_input_zip():
        path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
        if path:
            input_info['path'] = path
            input_info['type'] = 'ZIP'
            button_zip_input.config(text=path)
            button_dir_input.config(text="Välj mapp", state='disabled', style="TButton")
            toggle_buttons(button_zip_input, button_dir_input)
            # print(f"Selected input ZIP: {path}")

    def select_output_folder():
        path = filedialog.askdirectory()
        if path:
            output_path.set(path)
            button_output.config(text=path)
            # print(f"Selected output directory: {path}")

    def run_processing():
        if not input_info['path'] or not output_path.get():
            messagebox.showwarning("Varning", "Allt har inte valts")
            return
        try:
            # Before starting processing, backup existing files if any
            filenames = [
                'flow_region_mean_std_k1.txt',
                '48_region_mean_std_k1.txt',
                'mean_and_std_first_values_yz_neg_x0_k1.npy',
                'mean_and_std_first_values_yz_pos_x1_k1.npy',
                'mean_and_std_first_values_yz_pos_x2_k1.npy',
                'mean_and_std_first_values_yz_neg_x2_k1.npy',
                'mean_and_std_first_values_xz_pos_y2_k1.npy',
                'mean_and_std_first_values_xz_neg_y2_k1.npy',
            ]
            backup_existing_files(output_path.get(), filenames)
    
            # Show "Beräknar..." message
            calculating_label.config(text="Beräknar...")
            root.update()
            # Call your processing function
            water_mean_std(input_info['path'], output_path.get())
            # Hide "Beräknar..." message
            calculating_label.config(text="")
            messagebox.showinfo("Klart")
            back_to_main(root)
        except Exception as e:
            calculating_label.config(text="")
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Input buttons
    input_frame = tk.Frame(main_frame, bg="black")
    input_frame.pack(pady=5)
    button_dir_input = ttk.Button(input_frame, text="Välj mapp", command=select_input_directory)
    button_zip_input = ttk.Button(input_frame, text="Välj zip", command=select_input_zip)

    button_dir_input.pack(side='left', padx=5)
    button_zip_input.pack(side='left', padx=5)

    # Output button
    button_output = ttk.Button(main_frame, text="Välj mapp för resultat", command=select_output_folder)
    button_output.pack(pady=10)

    # Calculating label
    calculating_label = tk.Label(main_frame, text="", bg="black", fg="white", font=("Arial", 12))
    calculating_label.pack(pady=5)

    # Run button with green outline
    frame_run = tk.Frame(main_frame, highlightbackground="green", highlightthickness=2)
    frame_run.pack(pady=10)
    button_run = ttk.Button(frame_run, text='Starta', command=run_processing, style="Run.TButton")
    button_run.pack()

    # Back button
    button_back = ttk.Button(main_frame, text='Tillbaka', command=lambda: back_to_main(root))
    button_back.pack(pady=5)


def process_wat(root):
    """
    Processing window for wat2/wat1.
    Allows selecting two inputs (each can be folder or ZIP) and an output folder.
    """
    clear_window(root)

    # Create a main frame
    main_frame = tk.Frame(root, bg="black", padx=10, pady=10)
    main_frame.pack(expand=True, fill='both')

    # Add label
    label = tk.Label(main_frame, text="O-15 Vatten (Stress/Basline)", bg="black", fg="white", font=("Arial", 14))
    label.pack(pady=10)

    # Input information
    input_info1 = {'path': None, 'type': None}
    input_info2 = {'path': None, 'type': None}
    output_path = tk.StringVar()

    # Define button toggle function for first input
    def toggle_buttons_active1(active_button, inactive_button):
        inactive_button.config(state="disabled")
        # active_button remains enabled

    # Define button toggle function for second input
    def toggle_buttons_active2(active_button, inactive_button):
        inactive_button.config(state="disabled")
        # active_button remains enabled

    def select_input_directory1():
        path = filedialog.askdirectory()
        if path:
            input_info1['path'] = path
            input_info1['type'] = 'Directory'
            button_dir_input1.config(text=path)
            button_zip_input1.config(text="Välj zip 1", state='disabled', style="TButton")
            toggle_buttons_active1(button_dir_input1, button_zip_input1)
            # print(f"Selected input directory 1: {path}")

    def select_input_zip1():
        path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
        if path:
            input_info1['path'] = path
            input_info1['type'] = 'ZIP'
            button_zip_input1.config(text=path)
            button_dir_input1.config(text="Välj mapp (Basline)", state='disabled', style="TButton")
            toggle_buttons_active1(button_zip_input1, button_dir_input1)
            # print(f"Selected input ZIP 1: {path}")

    def select_input_directory2():
        path = filedialog.askdirectory()
        if path:
            input_info2['path'] = path
            input_info2['type'] = 'Directory'
            button_dir_input2.config(text=path)
            button_zip_input2.config(text="Välj zip (Stress)", state='disabled', style="TButton")
            toggle_buttons_active2(button_dir_input2, button_zip_input2)
            # print(f"Selected input directory 2: {path}")

    def select_input_zip2():
        path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
        if path:
            input_info2['path'] = path
            input_info2['type'] = 'ZIP'
            button_zip_input2.config(text=path)
            button_dir_input2.config(text="Välj mapp (Stress)", state='disabled', style="TButton")
            toggle_buttons_active2(button_zip_input2, button_dir_input2)
            # print(f"Selected input ZIP 2: {path}")

    def select_output_folder():
        path = filedialog.askdirectory()
        if path:
            output_path.set(path)
            button_output.config(text=path)
            # print(f"Selected output directory: {path}")

    def run_processing():
        if not input_info1['path'] or not input_info2['path'] or not output_path.get():
            messagebox.showwarning("Varning", "Allt har inte valts")
            return
        try:
            # Before starting processing, backup existing files if any
            filenames = [
                # Files generated by wat1_wat2_mean_std
                'flow_region_mean_std_k1_1.txt',
                '48_region_mean_std_k1_1.txt',
                'mean_and_std_first_values_yz_neg_x0_k1_1.npy',
                'mean_and_std_first_values_yz_pos_x1_k1_1.npy',
                'mean_and_std_first_values_yz_pos_x2_k1_1.npy',
                'mean_and_std_first_values_yz_neg_x2_k1_1.npy',
                'mean_and_std_first_values_xz_pos_y2_k1_1.npy',
                'mean_and_std_first_values_xz_neg_y2_k1_1.npy',
                'flow_region_mean_std_k1_1_ref.txt',
                '48_region_mean_std_k1_1_ref.txt',
                'mean_and_std_first_values_yz_neg_x0_k1_1_ref.npy',
                'mean_and_std_first_values_yz_pos_x1_k1_1_ref.npy',
                'mean_and_std_first_values_yz_pos_x2_k1_1_ref.npy',
                'mean_and_std_first_values_yz_neg_x2_k1_1_ref.npy',
                'mean_and_std_first_values_xz_pos_y2_k1_1_ref.npy',
                'mean_and_std_first_values_xz_neg_y2_k1_1_ref.npy',
                'flow_region_mean_std_k1_2.txt',
                '48_region_mean_std_k1_2.txt',
                'mean_and_std_first_values_yz_neg_x0_k1_2.npy',
                'mean_and_std_first_values_yz_pos_x1_k1_2.npy',
                'mean_and_std_first_values_yz_pos_x2_k1_2.npy',
                'mean_and_std_first_values_yz_neg_x2_k1_2.npy',
                'mean_and_std_first_values_xz_pos_y2_k1_2.npy',
                'mean_and_std_first_values_xz_neg_y2_k1_2.npy',
                'flow_region_mean_std_k1_2_ref.txt',
                '48_region_mean_std_k1_2_ref.txt',
                'mean_and_std_first_values_yz_neg_x0_k1_2_ref.npy',
                'mean_and_std_first_values_yz_pos_x1_k1_2_ref.npy',
                'mean_and_std_first_values_yz_pos_x2_k1_2_ref.npy',
                'mean_and_std_first_values_yz_neg_x2_k1_2_ref.npy',
                'mean_and_std_first_values_xz_pos_y2_k1_2_ref.npy',
                'mean_and_std_first_values_xz_neg_y2_k1_2_ref.npy',
                'flow_region_mean_std_k1_3.txt',
                '48_region_mean_std_k1_3.txt',
                'mean_and_std_first_values_yz_neg_x0_k1_3.npy',
                'mean_and_std_first_values_yz_pos_x1_k1_3.npy',
                'mean_and_std_first_values_yz_pos_x2_k1_3.npy',
                'mean_and_std_first_values_yz_neg_x2_k1_3.npy',
                'mean_and_std_first_values_xz_pos_y2_k1_3.npy',
                'mean_and_std_first_values_xz_neg_y2_k1_3.npy',
                'flow_region_mean_std_k1_3_ref.txt',
                '48_region_mean_std_k1_3_ref.txt',
                'mean_and_std_first_values_yz_neg_x0_k1_3_ref.npy',
                'mean_and_std_first_values_yz_pos_x1_k1_3_ref.npy',
                'mean_and_std_first_values_yz_pos_x2_k1_3_ref.npy',
                'mean_and_std_first_values_yz_neg_x2_k1_3_ref.npy',
                'mean_and_std_first_values_xz_pos_y2_k1_3_ref.npy',
                'mean_and_std_first_values_xz_neg_y2_k1_3_ref.npy',
            ]
            backup_existing_files(output_path.get(), filenames)
    
            # Show "Beräknar..." message
            calculating_label.config(text="Beräknar...")
            root.update()
            # Call your processing function
            wat1_wat2_mean_std(input_info1['path'], input_info2['path'], output_path.get())
            # Hide "Beräknar..." message
            calculating_label.config(text="")
            messagebox.showinfo("Klart")
            back_to_main(root)
        except Exception as e:
            calculating_label.config(text="")
            messagebox.showerror("Error", f"An error occurred: {e}")


    # Frame for first set of input buttons
    input_frame1 = tk.Frame(main_frame, bg="black")
    input_frame1.pack(pady=5)

    # Input buttons for first input
    button_dir_input1 = ttk.Button(input_frame1, text="Välj mapp (Basline)", command=select_input_directory1)
    button_zip_input1 = ttk.Button(input_frame1, text="Välj zip (Basline)", command=select_input_zip1)

    # Place the buttons on the same row
    button_dir_input1.pack(side='left', padx=5)
    button_zip_input1.pack(side='left', padx=5)

    # Frame for second set of input buttons
    input_frame2 = tk.Frame(main_frame, bg="black")
    input_frame2.pack(pady=5)

    # Input buttons for second input
    button_dir_input2 = ttk.Button(input_frame2, text="Välj mapp (Stress)", command=select_input_directory2)
    button_zip_input2 = ttk.Button(input_frame2, text="Välj zip (Stress)", command=select_input_zip2)

    # Place the buttons on the same row
    button_dir_input2.pack(side='left', padx=5)
    button_zip_input2.pack(side='left', padx=5)

    # Output button
    button_output = ttk.Button(main_frame, text="Välj mapp för resultat", command=select_output_folder)
    button_output.pack(pady=10)

    # Calculating label
    calculating_label = tk.Label(main_frame, text="", bg="black", fg="white", font=("Arial", 12))
    calculating_label.pack(pady=5)

    # Run button with green outline
    frame_run = tk.Frame(main_frame, highlightbackground="green", highlightthickness=2)
    frame_run.pack(pady=10)
    button_run = ttk.Button(frame_run, text='Starta', command=run_processing, style="Run.TButton")
    button_run.pack()

    # Back button
    button_back = ttk.Button(main_frame, text='Tillbaka', command=lambda: back_to_main(root))
    button_back.pack(pady=5)


def process_F18(root):
    """
    Processing window for F-18.
    Allows selecting one input (folder or ZIP) and an output folder.
    """
    clear_window(root)

    # Create a main frame
    main_frame = tk.Frame(root, bg="black", padx=10, pady=10)
    main_frame.pack(expand=True, fill='both')

    # Add label
    label = tk.Label(main_frame, text="F-18 PE2I", bg="black", fg="white", font=("Arial", 14))
    label.pack(pady=10)

    # Input information
    input_info = {'path': None, 'type': None}
    output_path = tk.StringVar()

    # Define button toggle function
    def toggle_buttons(active_button, inactive_button):
        inactive_button.config(state="disabled")
        # active_button remains enabled

    def select_input_directory():
        path = filedialog.askdirectory()
        if path:
            input_info['path'] = path
            input_info['type'] = 'Directory'
            button_dir_input.config(text=path)
            button_zip_input.config(text="Välj zip", state='disabled', style="TButton")
            toggle_buttons(button_dir_input, button_zip_input)
            # print(f"Selected input directory: {path}")

    def select_input_zip():
        path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
        if path:
            input_info['path'] = path
            input_info['type'] = 'ZIP'
            button_zip_input.config(text=path)
            button_dir_input.config(text="Välj mapp", state='disabled', style="TButton")
            toggle_buttons(button_zip_input, button_dir_input)
            # print(f"Selected input ZIP: {path}")

    def select_output_folder():
        path = filedialog.askdirectory()
        if path:
            output_path.set(path)
            button_output.config(text=path)
            # print(f"Selected output directory: {path}")

    def run_processing():
        if not input_info['path'] or not output_path.get():
            messagebox.showwarning("Varning", "Allt har inte valts")
            return
        try:
            # Before starting processing, backup existing files if any
            filenames = [
                'flow_region_mean_std_R_I.txt',
                '48_region_mean_std_R_I.txt',
            ]
            backup_existing_files(output_path.get(), filenames)
    
            # Show "Beräknar..." message
            calculating_label.config(text="Beräknar...")
            root.update()
            # Call your processing function
            R_I_mean_std(input_info['path'], output_path.get())
            # Hide "Beräknar..." message
            calculating_label.config(text="")
            messagebox.showinfo("Klart", "Processing completed successfully.")
            back_to_main(root)
        except Exception as e:
            calculating_label.config(text="")
            messagebox.showerror("Error", f"An error occurred: {e}")


    # Input buttons
    input_frame = tk.Frame(main_frame, bg="black")
    input_frame.pack(pady=5)
    button_dir_input = ttk.Button(input_frame, text="Välj mapp", command=select_input_directory)
    button_zip_input = ttk.Button(input_frame, text="Välj zip", command=select_input_zip)

    button_dir_input.pack(side='left', padx=5)
    button_zip_input.pack(side='left', padx=5)

    # Output button
    button_output = ttk.Button(main_frame, text="Välj mapp för resultat", command=select_output_folder)
    button_output.pack(pady=10)

    # Calculating label
    calculating_label = tk.Label(main_frame, text="", bg="black", fg="white", font=("Arial", 12))
    calculating_label.pack(pady=5)

    # Run button with green outline
    frame_run = tk.Frame(main_frame, highlightbackground="green", highlightthickness=2)
    frame_run.pack(pady=10)
    button_run = ttk.Button(frame_run, text='Starta', command=run_processing, style="Run.TButton")
    button_run.pack()

    # Back button
    button_back = ttk.Button(main_frame, text='Tillbaka', command=lambda: back_to_main(root))
    button_back.pack(pady=5)


def process_C11(root):
    """
    Processing window for C-11.
    Allows selecting one input (folder or ZIP) and an output folder.
    """
    clear_window(root)

    # Create a main frame
    main_frame = tk.Frame(root, bg="black", padx=10, pady=10)
    main_frame.pack(expand=True, fill='both')

    # Add label
    label = tk.Label(main_frame, text="C-11 PE2I", bg="black", fg="white", font=("Arial", 14))
    label.pack(pady=10)

    # Input information
    input_info = {'path': None, 'type': None}
    output_path = tk.StringVar()

    # Define button toggle function
    def toggle_buttons(active_button, inactive_button):
        inactive_button.config(state="disabled")
        # active_button remains enabled

    def select_input_directory():
        path = filedialog.askdirectory()
        if path:
            input_info['path'] = path
            input_info['type'] = 'Directory'
            button_dir_input.config(text=path)
            button_zip_input.config(text="Välj zip", state='disabled', style="TButton")
            toggle_buttons(button_dir_input, button_zip_input)
            # print(f"Selected input directory: {path}")

    def select_input_zip():
        path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
        if path:
            input_info['path'] = path
            input_info['type'] = 'ZIP'
            button_zip_input.config(text=path)
            button_dir_input.config(text="Välj mapp", state='disabled', style="TButton")
            toggle_buttons(button_zip_input, button_dir_input)
            # print(f"Selected input ZIP: {path}")

    def select_output_folder():
        path = filedialog.askdirectory()
        if path:
            output_path.set(path)
            button_output.config(text=path)
            # print(f"Selected output directory: {path}")

    def run_processing():
        if not input_info['path'] or not output_path.get():
            messagebox.showwarning("Varning", "Allt har inte valts")
            return
        try:
            # Before starting processing, backup existing files if any
            filenames = [
                'flow_region_mean_std_R_I_C.txt',
                '48_region_mean_std_R_I_C.txt',
                # Add any other filenames generated by R_I_C_mean_std
            ]
            backup_existing_files(output_path.get(), filenames)
    
            # Show "Beräknar..." message
            calculating_label.config(text="Beräknar...")
            root.update()
            # Call your processing function
            R_I_C_mean_std(input_info['path'], output_path.get())
            # Hide "Beräknar..." message
            calculating_label.config(text="")
            messagebox.showinfo("Klart", "Processing completed successfully.")
            back_to_main(root)
        except Exception as e:
            calculating_label.config(text="")
            messagebox.showerror("Error", f"An error occurred: {e}")


    # Input buttons
    input_frame = tk.Frame(main_frame, bg="black")
    input_frame.pack(pady=5)
    button_dir_input = ttk.Button(input_frame, text="Välj mapp", command=select_input_directory)
    button_zip_input = ttk.Button(input_frame, text="Välj zip", command=select_input_zip)

    button_dir_input.pack(side='left', padx=5)
    button_zip_input.pack(side='left', padx=5)

    # Output button
    button_output = ttk.Button(main_frame, text="Välj mapp för resultat", command=select_output_folder)
    button_output.pack(pady=10)

    # Calculating label
    calculating_label = tk.Label(main_frame, text="", bg="black", fg="white", font=("Arial", 12))
    calculating_label.pack(pady=5)

    # Run button with green outline
    frame_run = tk.Frame(main_frame, highlightbackground="green", highlightthickness=2)
    frame_run.pack(pady=10)
    button_run = ttk.Button(frame_run, text='Starta', command=run_processing, style="Run.TButton")
    button_run.pack()

    # Back button
    button_back = ttk.Button(main_frame, text='Tillbaka', command=lambda: back_to_main(root))
    button_back.pack(pady=5)


def back_to_main(root):
    """
    Returns to the main menu by clearing the current window and recreating the main frame.
    """
    clear_window(root)

    # Create a main frame
    main_frame = tk.Frame(root, bg="black", padx=10, pady=10)
    main_frame.pack(expand=True, fill='both')

    # Add label
    label = tk.Label(main_frame, text="Välj vilken du vill skapa nytt normalmaterial för",
                     bg="black", fg="white", font=("Arial", 14))
    label.pack(pady=20)

    # Create a frame for the buttons
    button_frame = tk.Frame(main_frame, bg="black")
    button_frame.pack(pady=10)

    # Create four buttons
    button_k1 = ttk.Button(button_frame, text="O-15", command=lambda: process_k1(root))
    button_wat = ttk.Button(button_frame, text="O-15 (stress/basline)", command=lambda: process_wat(root))
    button_F18 = ttk.Button(button_frame, text="F-18", command=lambda: process_F18(root))
    button_C11 = ttk.Button(button_frame, text="C-11", command=lambda: process_C11(root))

    # Place the buttons
    button_k1.grid(row=0, column=0, padx=10, pady=10)
    button_wat.grid(row=0, column=1, padx=10, pady=10)
    button_F18.grid(row=0, column=2, padx=10, pady=10)
    button_C11.grid(row=0, column=3, padx=10, pady=10)


def clear_window(root):
    """
    Clears all widgets from the root window.
    """
    for widget in root.winfo_children():
        widget.destroy()


try:
    # Your main code goes here
    if __name__ == "__main__":
        main()
except Exception as e:
    # Write the traceback to the temp file
    traceback.print_exc(file=temp_output)
    # Optionally, you can print a message
    print(f"An error occurred: {e}")
finally:
    # Close the temp file
    temp_output.close()
    # Delete the temp file if no exceptions occurred
    if 'e' not in locals():
        os.remove(temp_output_path)
