"""
BraTS 2020 Data Exploration Script (HDF5 Format)

This script loads and visualizes BraTS 2020 case data from .h5 files including:
- T1-weighted images
- T2-weighted images
- FLAIR images
- T1CE images (T1 with contrast enhancement)
- Segmentation masks

The data is stored as individual slice files (volume_X_slice_Y.h5) that need to be
reconstructed into full 3D volumes.

It performs basic data exploration including shape verification,
alignment checks, and visualization of multiple slices.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def load_brats_volume(data_dir, volume_number):
    """
    Load all imaging modalities and segmentation mask for a BraTS volume.
    
    The data is stored as individual .h5 files per slice. Each file contains:
    - 'image': 4-channel array (240, 240, 4) with modalities [T1, T2, FLAIR, T1CE]
    - 'mask': 3-channel array (240, 240, 3) with segmentation labels
    
    Args:
        data_dir: Path to the directory containing the .h5 files
        volume_number: Volume number to load (e.g., 9, 41, 99)
        
    Returns:
        Dictionary containing loaded volumes (3D arrays)
    """
    data_path = Path(data_dir)
    
    # Find all slice files for this volume
    pattern = f"volume_{volume_number}_slice_*.h5"
    slice_files = sorted(data_path.glob(pattern), 
                        key=lambda x: int(re.search(r'slice_(\d+)', x.name).group(1)))
    
    # If no files found, check in 'data' subdirectory (common structure)
    if not slice_files:
        data_subdir = data_path / "data"
        if data_subdir.exists():
            slice_files = sorted(data_subdir.glob(pattern), 
                                key=lambda x: int(re.search(r'slice_(\d+)', x.name).group(1)))
            if slice_files:
                print(f"Found files in subdirectory: {data_subdir}")
                data_path = data_subdir
    
    if not slice_files:
        raise FileNotFoundError(
            f"No slice files found for volume {volume_number}. "
            f"Expected pattern: volume_{volume_number}_slice_*.h5\n"
            f"Searched in: {data_path}\n"
            f"Also checked: {data_path / 'data' if (data_path / 'data').exists() else 'N/A'}"
        )
    
    print(f"Found {len(slice_files)} slices for volume {volume_number}")
    
    # Load first slice to get dimensions
    with h5py.File(slice_files[0], 'r') as f:
        slice_shape = f['image'].shape[:2]  # (240, 240)
        n_channels = f['image'].shape[2]  # Should be 4
    
    n_slices = len(slice_files)
    
    # Initialize arrays for full volumes
    # Image has 4 channels: [T1, T2, FLAIR, T1CE] (based on typical BraTS order)
    volumes = {
        'T1': np.zeros((*slice_shape, n_slices)),
        'T2': np.zeros((*slice_shape, n_slices)),
        'FLAIR': np.zeros((*slice_shape, n_slices)),
        'T1CE': np.zeros((*slice_shape, n_slices)),
        'Segmentation': np.zeros((*slice_shape, n_slices), dtype=np.uint8)
    }
    
    # Load each slice and stack them
    print("Loading slices...")
    for slice_idx, slice_file in enumerate(slice_files):
        with h5py.File(slice_file, 'r') as f:
            # Extract image data (shape: 240, 240, 4)
            image_data = f['image'][:]
            
            # Extract modalities from channels
            # Note: Channel order may vary. Common order is [T1, T2, FLAIR, T1CE]
            volumes['T1'][:, :, slice_idx] = image_data[:, :, 0]
            volumes['T2'][:, :, slice_idx] = image_data[:, :, 1]
            volumes['FLAIR'][:, :, slice_idx] = image_data[:, :, 2]
            volumes['T1CE'][:, :, slice_idx] = image_data[:, :, 3]
            
            # Extract mask data (shape: 240, 240, 3)
            # The mask has 3 binary channels representing different tumor regions:
            # Channel 0: label0 (typically NCR/NET - label 1 in BraTS)
            # Channel 1: label1 (typically ED - label 2 in BraTS)
            # Channel 2: label2 (typically ET - label 4 in BraTS)
            mask_data = f['mask'][:]
            
            # Combine channels into standard BraTS label encoding:
            # 0: Background
            # 1: Necrotic and non-enhancing tumor (NCR/NET) - from channel 0
            # 2: Peritumoral edema (ED) - from channel 1
            # 4: Enhancing tumor (ET) - from channel 2
            combined_mask = np.zeros(mask_data.shape[:2], dtype=np.uint8)
            combined_mask[mask_data[:, :, 0] > 0] = 1  # NCR/NET
            combined_mask[mask_data[:, :, 1] > 0] = 2  # ED
            combined_mask[mask_data[:, :, 2] > 0] = 4  # ET
            
            volumes['Segmentation'][:, :, slice_idx] = combined_mask
        
        if (slice_idx + 1) % 20 == 0:
            print(f"  Loaded {slice_idx + 1}/{n_slices} slices...")
    
    print(f"Successfully loaded volume {volume_number} with {n_slices} slices")
    return volumes


def print_volume_shapes(volumes):
    """
    Print the shape of each loaded volume.
    
    Args:
        volumes: Dictionary containing volume arrays
    """
    print("\n" + "="*60)
    print("VOLUME SHAPES")
    print("="*60)
    for modality, volume in volumes.items():
        print(f"{modality:15s}: {volume.shape}")
    print("="*60 + "\n")


def verify_alignment(volumes):
    """
    Verify that all volumes have the same spatial dimensions.
    
    Since HDF5 files don't contain affine transformation matrices,
    we only check that all volumes have the same shape.
    
    Args:
        volumes: Dictionary containing volume arrays
        
    Returns:
        Boolean indicating if all volumes are aligned
    """
    print("="*60)
    print("ALIGNMENT VERIFICATION")
    print("="*60)
    
    # Check if all volumes have the same shape
    shapes = [vol.shape for vol in volumes.values()]
    all_same_shape = all(shape == shapes[0] for shape in shapes)
    
    if all_same_shape:
        print(f"✓ All volumes have the same shape: {shapes[0]}")
        print(f"  Spatial dimensions: {shapes[0][:2]}")
        print(f"  Number of slices: {shapes[0][2]}")
    else:
        print("✗ Volumes have different shapes:")
        for modality, shape in zip(volumes.keys(), shapes):
            print(f"  {modality}: {shape}")
        return False
    
    print("="*60 + "\n")
    return True


def display_slices(volumes, n_slices=3):
    """
    Display multiple axial slices for each modality.
    
    Args:
        volumes: Dictionary containing volume arrays
        n_slices: Number of slices to display per modality
    """
    # Get the depth (z-axis) dimension
    depth = volumes['T1'].shape[2]
    
    # Select evenly spaced slices
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, n_slices, dtype=int)
    
    # Create figure with subplots
    # Exclude segmentation from the main display (it will be shown separately)
    display_modalities = {k: v for k, v in volumes.items() if k != 'Segmentation'}
    
    fig, axes = plt.subplots(len(display_modalities), n_slices, 
                             figsize=(4*n_slices, 4*len(display_modalities)))
    
    # Handle case where we only have one modality
    if len(display_modalities) == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each modality
    for row_idx, (modality, volume) in enumerate(display_modalities.items()):
        for col_idx, slice_idx in enumerate(slice_indices):
            ax = axes[row_idx, col_idx]
            
            # Extract axial slice (z-axis is the third dimension)
            slice_data = volume[:, :, slice_idx]
            
            # Display the slice
            im = ax.imshow(slice_data, cmap='gray', origin='lower')
            ax.set_title(f'{modality}\nSlice {slice_idx}/{depth-1}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.suptitle('Axial Slices - All Modalities', y=1.02, fontsize=14, fontweight='bold')
    plt.show()


def overlay_segmentation(flair_volume, seg_volume, slice_idx=None):
    """
    Overlay segmentation mask on FLAIR image.
    
    Args:
        flair_volume: FLAIR volume array
        seg_volume: Segmentation mask volume array
        slice_idx: Index of slice to display (default: middle slice)
    """
    if slice_idx is None:
        slice_idx = flair_volume.shape[2] // 2
    
    # Extract axial slice
    flair_slice = flair_volume[:, :, slice_idx]
    seg_slice = seg_volume[:, :, slice_idx]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display FLAIR only
    axes[0].imshow(flair_slice, cmap='gray', origin='lower')
    axes[0].set_title(f'FLAIR - Slice {slice_idx}', fontsize=12)
    axes[0].axis('off')
    
    # Display FLAIR with segmentation overlay
    axes[1].imshow(flair_slice, cmap='gray', origin='lower', alpha=0.7)
    
    # Create colored overlay for segmentation
    # BraTS segmentation labels:
    # 0: Background
    # 1: Necrotic and non-enhancing tumor (NCR/NET)
    # 2: Peritumoral edema (ED)
    # 4: Enhancing tumor (ET)
    # Combined labels: 1+2+4 = Complete tumor, 1+4 = Tumor core, 4 = Enhancing core
    
    # Note: The mask in HDF5 format may use different encoding.
    # We'll check what values are present and map them appropriately.
    unique_values = np.unique(seg_slice)
    print(f"\nUnique segmentation values in slice {slice_idx}: {unique_values}")
    
    # Create RGB overlay
    overlay = np.zeros((*seg_slice.shape, 3))
    
    # Color coding:
    # Red: Necrotic and non-enhancing tumor (label 1)
    # Green: Peritumoral edema (label 2)
    # Blue: Enhancing tumor (label 4)
    
    # Handle different possible label encodings
    if 1 in unique_values:
        overlay[seg_slice == 1] = [1, 0, 0]  # Red for NCR/NET
    if 2 in unique_values:
        overlay[seg_slice == 2] = [0, 1, 0]  # Green for ED
    if 4 in unique_values:
        overlay[seg_slice == 4] = [0, 0, 1]  # Blue for ET
    
    # If mask uses binary encoding (0/1), show it in red
    if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
        overlay[seg_slice == 1] = [1, 0, 0]  # Red for tumor
    
    # Display overlay
    axes[1].imshow(overlay, alpha=0.5, origin='lower')
    axes[1].set_title(f'FLAIR + Segmentation Overlay\nSlice {slice_idx}', fontsize=12)
    axes[1].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = []
    if 1 in unique_values:
        legend_elements.append(Patch(facecolor='red', alpha=0.5, label='NCR/NET (Label 1)'))
    if 2 in unique_values:
        legend_elements.append(Patch(facecolor='green', alpha=0.5, label='ED (Label 2)'))
    if 4 in unique_values:
        legend_elements.append(Patch(facecolor='blue', alpha=0.5, label='ET (Label 4)'))
    if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
        legend_elements = [Patch(facecolor='red', alpha=0.5, label='Tumor (Label 1)')]
    
    if legend_elements:
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print segmentation statistics for this slice
    unique_labels, counts = np.unique(seg_slice, return_counts=True)
    print(f"\nSegmentation statistics for slice {slice_idx}:")
    label_names = {0: 'Background', 1: 'NCR/NET', 2: 'ED', 4: 'ET'}
    for label, count in zip(unique_labels, counts):
        percentage = (count / seg_slice.size) * 100
        label_name = label_names.get(int(label), f'Label {int(label)}')
        print(f"  {label_name}: {count} voxels ({percentage:.2f}%)")


def main():
    """
    Main function to run the BraTS data exploration.
    """
    # Get data directory and volume number
    data_directory = input("Enter the path to the data directory containing .h5 files: ").strip()
    data_directory = data_directory.strip('"').strip("'")
    
    volume_input = input("Enter the volume number to explore (e.g., 9, 41, 99): ").strip()
    try:
        volume_number = int(volume_input)
    except ValueError:
        print(f"Error: '{volume_input}' is not a valid volume number")
        return
    
    print(f"\nExploring BraTS volume {volume_number} in: {data_directory}\n")
    
    try:
        # Load all volumes
        volumes = load_brats_volume(data_directory, volume_number)
        
        # Print volume shapes
        print_volume_shapes(volumes)
        
        # Verify alignment
        is_aligned = verify_alignment(volumes)
        
        if not is_aligned:
            print("WARNING: Volumes may not be properly aligned!")
            print("Proceeding with visualization anyway...\n")
        
        # Display slices for each modality
        print("Displaying 3 axial slices for each modality...")
        display_slices(volumes, n_slices=3)
        
        # Overlay segmentation on FLAIR
        print("\nDisplaying segmentation overlay on FLAIR...")
        overlay_segmentation(volumes['FLAIR'], volumes['Segmentation'])
        
        # Additional statistics
        print("\n" + "="*60)
        print("ADDITIONAL STATISTICS")
        print("="*60)
        for modality, volume in volumes.items():
            if modality != 'Segmentation':
                print(f"\n{modality}:")
                print(f"  Min value: {volume.min():.2f}")
                print(f"  Max value: {volume.max():.2f}")
                print(f"  Mean value: {volume.mean():.2f}")
                print(f"  Std value: {volume.std():.2f}")
        
        print("\n" + "="*60)
        print("Segmentation:")
        unique_labels, counts = np.unique(volumes['Segmentation'], return_counts=True)
        label_names = {0: 'Background', 1: 'NCR/NET', 2: 'ED', 4: 'ET'}
        for label, count in zip(unique_labels, counts):
            percentage = (count / volumes['Segmentation'].size) * 100
            label_name = label_names.get(int(label), f'Label {int(label)}')
            print(f"  {label_name}: {count} voxels ({percentage:.2f}%)")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. The data directory path is correct")
        print("2. The directory contains .h5 files with pattern: volume_X_slice_Y.h5")
        print("3. The volume number exists in the dataset")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
