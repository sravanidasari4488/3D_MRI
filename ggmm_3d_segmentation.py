"""
3D Generalized Gaussian Mixture Model (GGMM) Segmentation

This module extends the 2D GGMM segmentation to process full 3D MRI volumes.
It processes each slice independently using the 2D GGMM and combines the
results into a 3D segmented volume.

The processing is memory-safe and includes progress tracking for large volumes.
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from ggmm_segmentation import segment_image, GeneralizedGaussianMixtureModel
from sklearn.preprocessing import MinMaxScaler
import gc  # For memory management


def load_nifti_volume(nifti_path):
    """
    Load a 3D NIfTI volume.
    
    Parameters:
    -----------
    nifti_path : str or Path
        Path to the NIfTI file (.nii or .nii.gz)
    
    Returns:
    --------
    volume : ndarray
        3D volume array
    nifti_img : nibabel.Nifti1Image
        NIfTI image object (for metadata)
    """
    nifti_path = Path(nifti_path)
    
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
    
    print(f"Loading NIfTI volume: {nifti_path.name}")
    nifti_img = nib.load(str(nifti_path))
    volume = nifti_img.get_fdata()
    
    print(f"Volume shape: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print(f"Intensity range: [{volume.min():.4f}, {volume.max():.4f}]")
    
    return volume, nifti_img


def apply_ggmm_to_slice(slice_2d, n_components=3, model=None):
    """
    Apply GGMM segmentation to a single 2D slice.
    
    This is a wrapper around the segment_image function that allows
    reusing a pre-initialized model or creating a new one.
    
    Parameters:
    -----------
    slice_2d : ndarray, shape (height, width)
        2D grayscale slice
    n_components : int
        Number of clusters
    model : GeneralizedGaussianMixtureModel, optional
        Pre-initialized model (if None, creates new model)
    
    Returns:
    --------
    segmented_slice : ndarray, shape (height, width)
        Segmented slice with cluster labels
    model : GeneralizedGaussianMixtureModel
        Fitted model (reused or newly created)
    """
    # Flatten and normalize
    original_shape = slice_2d.shape
    flattened = slice_2d.flatten()
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(flattened.reshape(-1, 1)).flatten()
    
    # Use existing model or create new one
    if model is None:
        model = GeneralizedGaussianMixtureModel(n_components=n_components, 
                                               beta=2.0, max_iter=100, tol=1e-6)
        model.fit(normalized)
    else:
        # Re-fit model for this slice (each slice may have different intensity distribution)
        model.fit(normalized)
    
    # Predict cluster assignments
    labels = model.predict(normalized)
    
    # Reshape back to original shape
    segmented_slice = labels.reshape(original_shape)
    
    return segmented_slice, model


def process_3d_volume_ggmm(volume, n_components=3, axis=2, verbose=True):
    """
    Process a full 3D volume using GGMM segmentation on each slice.
    
    Parameters:
    -----------
    volume : ndarray, shape (height, width, depth)
        3D MRI volume
    n_components : int
        Number of clusters for segmentation
    axis : int
        Axis along which to slice (default: 2 for z-axis)
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    segmented_volume : ndarray, shape (height, width, depth)
        3D segmented volume with cluster labels
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim}D array")
    
    # Get dimensions
    if axis == 0:
        n_slices = volume.shape[0]
        segmented_volume = np.zeros_like(volume, dtype=np.uint8)
    elif axis == 1:
        n_slices = volume.shape[1]
        segmented_volume = np.zeros_like(volume, dtype=np.uint8)
    elif axis == 2:
        n_slices = volume.shape[2]
        segmented_volume = np.zeros_like(volume, dtype=np.uint8)
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2")
    
    if verbose:
        print(f"\nProcessing {n_slices} slices along axis {axis}...")
        print("="*60)
    
    # Process each slice
    for slice_idx in range(n_slices):
        if verbose and (slice_idx % 10 == 0 or slice_idx == n_slices - 1):
            print(f"Processing slice {slice_idx + 1}/{n_slices}...", end='\r')
        
        # Extract slice
        if axis == 0:
            slice_2d = volume[slice_idx, :, :]
        elif axis == 1:
            slice_2d = volume[:, slice_idx, :]
        else:  # axis == 2
            slice_2d = volume[:, :, slice_idx]
        
        # Apply GGMM segmentation
        segmented_slice, _ = apply_ggmm_to_slice(slice_2d, n_components=n_components)
        
        # Store segmented slice
        if axis == 0:
            segmented_volume[slice_idx, :, :] = segmented_slice
        elif axis == 1:
            segmented_volume[:, slice_idx, :] = segmented_slice
        else:  # axis == 2
            segmented_volume[:, :, slice_idx] = segmented_slice
        
        # Memory management: periodically clear cache
        if slice_idx % 50 == 0 and slice_idx > 0:
            gc.collect()
    
    if verbose:
        print(f"\nCompleted processing all {n_slices} slices!")
        print("="*60)
    
    return segmented_volume


def save_segmented_volume(segmented_volume, output_path, reference_nifti=None, 
                         affine=None, header=None):
    """
    Save segmented volume as a NIfTI file.
    
    Parameters:
    -----------
    segmented_volume : ndarray
        3D segmented volume
    output_path : str or Path
        Output file path
    reference_nifti : nibabel.Nifti1Image, optional
        Reference NIfTI image to copy affine and header from
    affine : ndarray, optional
        Affine transformation matrix
    header : nibabel header, optional
        NIfTI header
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use reference image metadata if provided
    if reference_nifti is not None:
        affine = reference_nifti.affine
        header = reference_nifti.header.copy()
        header.set_data_dtype(np.uint8)
    elif affine is None:
        # Default affine (identity)
        affine = np.eye(4)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(segmented_volume.astype(np.uint8), affine, header)
    
    # Save
    print(f"\nSaving segmented volume to: {output_path}")
    nib.save(nifti_img, str(output_path))
    print(f"Saved successfully!")
    print(f"Output shape: {segmented_volume.shape}")
    print(f"Output dtype: {segmented_volume.dtype}")


def visualize_slice_comparison(original_volume, segmented_volume, slice_idx, axis=2):
    """
    Visualize comparison between original and segmented slice.
    
    Parameters:
    -----------
    original_volume : ndarray
        Original 3D volume
    segmented_volume : ndarray
        Segmented 3D volume
    slice_idx : int
        Index of slice to visualize
    axis : int
        Axis along which slice was taken
    """
    # Extract slices
    if axis == 0:
        original_slice = original_volume[slice_idx, :, :]
        segmented_slice = segmented_volume[slice_idx, :, :]
    elif axis == 1:
        original_slice = original_volume[:, slice_idx, :]
        segmented_slice = segmented_volume[:, slice_idx, :]
    else:  # axis == 2
        original_slice = original_volume[:, :, slice_idx]
        segmented_slice = segmented_volume[:, :, slice_idx]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original slice
    im1 = axes[0].imshow(original_slice, cmap='gray', origin='lower')
    axes[0].set_title(f'Original Slice {slice_idx}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Segmented slice
    im2 = axes[1].imshow(segmented_slice, cmap='viridis', origin='lower')
    axes[1].set_title(f'Segmented Slice {slice_idx}', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(original_slice, cmap='gray', origin='lower', alpha=0.7)
    im3 = axes[2].imshow(segmented_slice, cmap='viridis', origin='lower', 
                        alpha=0.5, interpolation='nearest')
    axes[2].set_title(f'Overlay: Slice {slice_idx}', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics for this slice
    print(f"\nSlice {slice_idx} Statistics:")
    print("-" * 40)
    unique_labels, counts = np.unique(segmented_slice, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = (count / segmented_slice.size) * 100
        print(f"  Cluster {label}: {count} pixels ({percentage:.2f}%)")


def process_3d_nifti_volume(input_path, output_path, n_components=3, 
                           axis=2, visualize_slice_idx=None):
    """
    Complete pipeline: Load 3D NIfTI, segment, save, and visualize.
    
    Parameters:
    -----------
    input_path : str or Path
        Path to input NIfTI file
    output_path : str or Path
        Path to output segmented NIfTI file
    n_components : int
        Number of clusters
    axis : int
        Axis along which to slice (default: 2)
    visualize_slice_idx : int, optional
        Index of slice to visualize (if None, uses middle slice)
    """
    print("="*60)
    print("3D GGMM SEGMENTATION PIPELINE")
    print("="*60)
    
    # Step 1: Load NIfTI volume
    print("\n[Step 1/4] Loading NIfTI volume...")
    volume, nifti_img = load_nifti_volume(input_path)
    
    # Step 2: Process 3D volume
    print(f"\n[Step 2/4] Processing 3D volume with {n_components} components...")
    segmented_volume = process_3d_volume_ggmm(volume, n_components=n_components, 
                                              axis=axis, verbose=True)
    
    # Step 3: Save segmented volume
    print(f"\n[Step 3/4] Saving segmented volume...")
    save_segmented_volume(segmented_volume, output_path, reference_nifti=nifti_img)
    
    # Step 4: Visualize
    if visualize_slice_idx is None:
        visualize_slice_idx = segmented_volume.shape[axis] // 2
    
    print(f"\n[Step 4/4] Visualizing slice {visualize_slice_idx}...")
    visualize_slice_comparison(volume, segmented_volume, visualize_slice_idx, axis=axis)
    
    # Print overall statistics
    print("\n" + "="*60)
    print("OVERALL SEGMENTATION STATISTICS")
    print("="*60)
    unique_labels, counts = np.unique(segmented_volume, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = (count / segmented_volume.size) * 100
        print(f"Cluster {label}: {count} voxels ({percentage:.2f}%)")
    print("="*60)
    
    return segmented_volume, nifti_img


def main():
    """
    Main function for 3D GGMM segmentation.
    """
    print("3D Generalized Gaussian Mixture Model Segmentation")
    print("="*60)
    
    # Get input file
    input_path = input("Enter path to input NIfTI file: ").strip()
    input_path = input_path.strip('"').strip("'")
    
    # Get output file
    output_path = input("Enter path for output segmented NIfTI file: ").strip()
    output_path = output_path.strip('"').strip("'")
    
    # Get number of components
    n_components_input = input("Enter number of clusters (default: 3): ").strip()
    n_components = int(n_components_input) if n_components_input else 3
    
    # Get slice axis
    axis_input = input("Enter slice axis (0, 1, or 2, default: 2): ").strip()
    axis = int(axis_input) if axis_input else 2
    
    # Get visualization slice
    viz_slice_input = input("Enter slice index to visualize (default: middle): ").strip()
    visualize_slice_idx = int(viz_slice_input) if viz_slice_input else None
    
    try:
        # Process the volume
        segmented_volume, nifti_img = process_3d_nifti_volume(
            input_path=input_path,
            output_path=output_path,
            n_components=n_components,
            axis=axis,
            visualize_slice_idx=visualize_slice_idx
        )
        
        print("\n✓ Processing complete!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("1. The input NIfTI file path is correct")
        print("2. The file exists and is readable")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

