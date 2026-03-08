"""
Generalized Gaussian Mixture Model (GGMM) for 2D Image Segmentation

This module implements a GGMM using the Expectation-Maximization (EM) algorithm
for segmenting 2D grayscale images. The model assumes each pixel intensity
follows a Generalized Gaussian distribution, allowing for flexible modeling
of different tissue types in medical images.

The EM algorithm iteratively:
1. E-step: Computes responsibilities (posterior probabilities) of each pixel
           belonging to each cluster
2. M-step: Updates model parameters (weights, means, scale parameters) to
           maximize the expected log-likelihood
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.special import gamma
import h5py
from pathlib import Path
import re
import warnings


def generalized_gaussian_pdf(x, mu, alpha, beta):
    """
    Compute the Generalized Gaussian Probability Density Function.
    
    Parameters:
    -----------
    x : array_like
        Input values at which to evaluate the PDF
    mu : float
        Location parameter (mean)
    alpha : float
        Scale parameter (controls spread)
    beta : float
        Shape parameter (beta = 2 for normal distribution)
    
    Returns:
    --------
    pdf : ndarray
        Probability density values at x
    """
    x = np.asarray(x, dtype=np.float64)
    mu = float(mu)
    alpha = float(alpha)
    beta = float(beta)
    
    if alpha <= 0 or beta <= 0:
        return np.zeros_like(x)
    
    # Compute absolute deviation and normalize
    abs_dev = np.abs(x - mu)
    normalized = abs_dev / alpha
    
    # Compute exponent with numerical stability
    exponent = np.power(normalized, beta)
    exponent = np.clip(exponent, 0, 700)
    exp_term = np.exp(-exponent)
    
    # Normalization constant
    normalization_constant = beta / (2.0 * alpha * gamma(1.0 / beta))
    
    # Compute PDF
    pdf = normalization_constant * exp_term
    
    # Handle numerical issues
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    
    return pdf


class GeneralizedGaussianMixtureModel:
    """
    Generalized Gaussian Mixture Model for image segmentation.
    
    The model assumes pixel intensities follow a mixture of K Generalized
    Gaussian distributions, where each component represents a different
    tissue type or region in the image.
    """
    
    def __init__(self, n_components=3, beta=2.0, max_iter=100, tol=1e-6):
        """
        Initialize the GGMM.
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components (clusters)
        beta : float
            Shape parameter for Generalized Gaussian (fixed)
            beta = 2 corresponds to normal distribution
        max_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence tolerance for log-likelihood change
        """
        self.n_components = n_components
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        
        # Model parameters (will be initialized in fit)
        self.weights_ = None      # Mixing weights (π_k)
        self.means_ = None         # Component means (μ_k)
        self.alphas_ = None        # Scale parameters (α_k)
        
        # For tracking convergence
        self.log_likelihoods_ = []
    
    def _initialize_parameters(self, X):
        """
        Initialize model parameters using KMeans clustering.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples,)
            Flattened and normalized pixel intensities
        """
        n_samples = len(X)
        
        # Check data variance - if too low, use fallback initialization
        data_std = np.std(X)
        data_range = np.max(X) - np.min(X)
        
        # Initialize means using KMeans with error handling
        try:
            # Suppress convergence warnings during initialization
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                kmeans = KMeans(n_clusters=self.n_components, n_init=10, 
                              random_state=42, max_iter=300)
                kmeans.fit(X.reshape(-1, 1))
                self.means_ = kmeans.cluster_centers_.flatten()
                labels = kmeans.labels_
        except Exception:
            # Fallback: use percentile-based initialization
            self.means_ = np.percentile(X, np.linspace(10, 90, self.n_components))
            # Assign labels based on nearest mean
            labels = np.argmin(np.abs(X[:, np.newaxis] - self.means_), axis=1)
        
        # Handle case where KMeans finds fewer clusters than requested
        unique_labels = np.unique(labels)
        if len(unique_labels) < self.n_components:
            # Fill missing clusters with evenly spaced means
            existing_means = self.means_[unique_labels]
            min_val, max_val = np.min(X), np.max(X)
            
            # Create evenly spaced means across the data range
            self.means_ = np.linspace(min_val, max_val, self.n_components)
            
            # Reassign labels based on new means
            labels = np.argmin(np.abs(X[:, np.newaxis] - self.means_), axis=1)
            unique_labels = np.unique(labels)
        
        # Initialize weights from cluster proportions
        self.weights_ = np.array([np.sum(labels == k) / n_samples 
                                  for k in range(self.n_components)])
        
        # Initialize alpha (scale parameter) using standard deviation within each cluster
        self.alphas_ = np.zeros(self.n_components)
        for k in range(self.n_components):
            cluster_data = X[labels == k]
            if len(cluster_data) > 1:
                # Use standard deviation as initial scale
                # For beta=2 (normal), alpha relates to std: std = alpha / sqrt(2)
                std = np.std(cluster_data)
                if std > 0:
                    self.alphas_[k] = std * np.sqrt(2)
                else:
                    # If no variance, use a fraction of the data range
                    self.alphas_[k] = max(data_range / (2 * self.n_components), 0.01)
            else:
                # Default: use fraction of data range
                self.alphas_[k] = max(data_range / (2 * self.n_components), 0.01)
        
        # Ensure weights sum to 1
        self.weights_ = self.weights_ / np.sum(self.weights_)
        
        # Ensure all weights are positive (avoid zero weights)
        self.weights_ = np.maximum(self.weights_, 1e-6)
        self.weights_ = self.weights_ / np.sum(self.weights_)
        
        # Only print if verbose mode (controlled by fit method)
        if hasattr(self, '_verbose') and self._verbose:
            print(f"Initialized {self.n_components} components:")
            for k in range(self.n_components):
                print(f"  Component {k}: weight={self.weights_[k]:.4f}, "
                      f"mean={self.means_[k]:.4f}, alpha={self.alphas_[k]:.4f}")
    
    def _e_step(self, X):
        """
        E-step: Compute responsibilities (posterior probabilities).
        
        For each pixel x_i and component k, compute:
        r_ik = P(z_i = k | x_i) = (π_k * p_k(x_i)) / Σ_j (π_j * p_j(x_i))
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples,)
            Flattened pixel intensities
        
        Returns:
        --------
        responsibilities : ndarray, shape (n_samples, n_components)
            Responsibility matrix
        """
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # Compute unnormalized responsibilities
        for k in range(self.n_components):
            # Compute likelihood: π_k * p_k(x_i)
            pdf_values = generalized_gaussian_pdf(X, self.means_[k], 
                                                  self.alphas_[k], self.beta)
            responsibilities[:, k] = self.weights_[k] * pdf_values
        
        # Normalize: sum over components should be 1 for each pixel
        row_sums = np.sum(responsibilities, axis=1)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        responsibilities = responsibilities / row_sums[:, np.newaxis]
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """
        M-step: Update model parameters to maximize expected log-likelihood.
        
        Updates:
        - Weights: π_k = (1/N) * Σ_i r_ik
        - Means: μ_k = Σ_i (r_ik * x_i) / Σ_i r_ik
        - Scale parameters: α_k (updated using moment-based estimation)
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples,)
            Flattened pixel intensities
        responsibilities : ndarray, shape (n_samples, n_components)
            Responsibility matrix from E-step
        """
        n_samples = len(X)
        
        # Update weights
        self.weights_ = np.sum(responsibilities, axis=0) / n_samples
        
        # Update means
        for k in range(self.n_components):
            sum_resp = np.sum(responsibilities[:, k])
            if sum_resp > 0:
                self.means_[k] = np.sum(responsibilities[:, k] * X) / sum_resp
            else:
                # If no responsibility, keep current mean
                pass
        
        # Update scale parameters (alpha)
        # For Generalized Gaussian, we use moment-based estimation
        # E[|X - μ|^β] = α^β * Γ(2/β) / Γ(1/β)
        for k in range(self.n_components):
            sum_resp = np.sum(responsibilities[:, k])
            if sum_resp > 0:
                # Compute weighted absolute deviations
                abs_dev = np.abs(X - self.means_[k])
                weighted_abs_dev_beta = np.sum(responsibilities[:, k] * 
                                               np.power(abs_dev, self.beta))
                
                # Estimate alpha using moment matching
                # E[|X - μ|^β] = α^β * Γ(2/β) / Γ(1/β)
                # Therefore: α^β = E[|X - μ|^β] * Γ(1/β) / Γ(2/β)
                expected_moment = weighted_abs_dev_beta / sum_resp
                
                if expected_moment > 0:
                    gamma_ratio = gamma(1.0 / self.beta) / gamma(2.0 / self.beta)
                    alpha_beta = expected_moment * gamma_ratio
                    self.alphas_[k] = np.power(alpha_beta, 1.0 / self.beta)
                else:
                    self.alphas_[k] = 0.1  # Default small value
                
                # Ensure alpha is positive and not too small
                self.alphas_[k] = max(self.alphas_[k], 1e-6)
            else:
                # If no responsibility, keep current alpha
                pass
        
        # Ensure weights sum to 1
        self.weights_ = self.weights_ / np.sum(self.weights_)
    
    def _compute_log_likelihood(self, X):
        """
        Compute the log-likelihood of the data under the current model.
        
        log L = Σ_i log(Σ_k π_k * p_k(x_i))
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples,)
            Flattened pixel intensities
        
        Returns:
        --------
        log_likelihood : float
            Log-likelihood value
        """
        # Vectorized implementation for efficiency
        # Compute pdf values for all components at once
        n_samples = len(X)
        if n_samples == 0:
            return 0.0

        # pdf_matrix[i, k] = p_k(x_i)
        pdf_matrix = np.empty((n_samples, self.n_components), dtype=np.float64)
        for k in range(self.n_components):
            pdf_matrix[:, k] = generalized_gaussian_pdf(
                X, self.means_[k], self.alphas_[k], self.beta
            )

        # Mixture density for each sample: sum_k π_k * p_k(x_i)
        mixture_density = np.dot(pdf_matrix, self.weights_)

        # Avoid log(0) by adding small epsilon
        eps = 1e-12
        mixture_density = np.clip(mixture_density, eps, None)

        log_likelihood = np.sum(np.log(mixture_density))
        return float(log_likelihood)
    
    def fit(self, X, verbose=True):
        """
        Fit the GGMM to the data using the EM algorithm.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples,)
            Flattened and normalized pixel intensities
        verbose : bool
            Whether to print progress information
        """
        self._verbose = verbose
        # Initialize parameters
        self._initialize_parameters(X)
        
        # EM iterations
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            responsibilities = self._e_step(X)
            
            # M-step: update parameters
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihoods_.append(log_likelihood)
            
            # Check convergence
            log_likelihood_change = log_likelihood - prev_log_likelihood
            
            if self._verbose:
                if iteration % 10 == 0 or iteration < 5:
                    print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.4f} "
                          f"(change = {log_likelihood_change:.6f})")
            
            if abs(log_likelihood_change) < self.tol:
                if self._verbose:
                    print(f"\nConverged after {iteration + 1} iterations!")
                    print(f"Final log-likelihood: {log_likelihood:.4f}")
                break
            
            prev_log_likelihood = log_likelihood
        
        if iteration == self.max_iter - 1:
            if self._verbose:
                print(f"\nReached maximum iterations ({self.max_iter})")
                print(f"Final log-likelihood: {log_likelihood:.4f}")
    
    def predict(self, X):
        """
        Predict cluster assignments for each pixel.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples,)
            Flattened pixel intensities
        
        Returns:
        --------
        labels : ndarray, shape (n_samples,)
            Cluster assignments (0 to n_components-1)
        """
        responsibilities = self._e_step(X)
        labels = np.argmax(responsibilities, axis=1)
        return labels
    
    def predict_proba(self, X):
        """
        Predict posterior probabilities (responsibilities) for each pixel.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples,)
            Flattened pixel intensities
        
        Returns:
        --------
        responsibilities : ndarray, shape (n_samples, n_components)
            Posterior probabilities
        """
        return self._e_step(X)


def load_mri_slice(data_dir, volume_number, slice_number):
    """
    Load a single MRI slice from BraTS dataset.
    
    Parameters:
    -----------
    data_dir : str
        Path to data directory
    volume_number : int
        Volume number
    slice_number : int
        Slice number within the volume
    
    Returns:
    --------
    slice_data : ndarray, shape (240, 240)
        Grayscale MRI slice (FLAIR modality)
    """
    data_path = Path(data_dir)
    
    # Check for data subdirectory
    if (data_path / "data").exists():
        data_path = data_path / "data"
    
    # Load slice file
    slice_file = data_path / f"volume_{volume_number}_slice_{slice_number}.h5"
    
    if not slice_file.exists():
        raise FileNotFoundError(f"Slice file not found: {slice_file}")
    
    with h5py.File(slice_file, 'r') as f:
        # Extract FLAIR channel (channel 2 in the 4-channel image)
        image_data = f['image'][:]
        slice_data = image_data[:, :, 2]  # FLAIR is channel 2
    
    return slice_data


def segment_image(image, n_components=3):
    """
    Segment a 2D grayscale image using GGMM.
    
    Parameters:
    -----------
    image : ndarray, shape (height, width)
        2D grayscale image
    n_components : int
        Number of segments (clusters)
    
    Returns:
    --------
    segmented_image : ndarray, shape (height, width)
        Segmented image with cluster labels
    model : GeneralizedGaussianMixtureModel
        Fitted GGMM model
    """
    # Step 1: Flatten image
    original_shape = image.shape
    flattened = image.flatten()
    
    # Step 2: Normalize using MinMaxScaler
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(flattened.reshape(-1, 1)).flatten()
    
    print(f"Image shape: {original_shape}")
    print(f"Flattened shape: {flattened.shape}")
    print(f"Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Step 3-6: Initialize and fit GGMM
    model = GeneralizedGaussianMixtureModel(n_components=n_components, 
                                           beta=2.0, max_iter=100, tol=1e-6)
    
    print("\nFitting GGMM...")
    model.fit(normalized, verbose=True)
    
    # Predict cluster assignments
    labels = model.predict(normalized)
    
    # Reshape back to original image shape
    segmented_image = labels.reshape(original_shape)
    
    return segmented_image, model, scaler


def visualize_segmentation(original_image, segmented_image, model=None):
    """
    Visualize original image and segmentation results.
    
    Parameters:
    -----------
    original_image : ndarray
        Original grayscale image
    segmented_image : ndarray
        Segmented image with cluster labels
    model : GeneralizedGaussianMixtureModel, optional
        Fitted model for displaying parameters
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    im1 = axes[0].imshow(original_image, cmap='gray', origin='lower')
    axes[0].set_title('Original MRI Slice (FLAIR)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Segmented image
    im2 = axes[1].imshow(segmented_image, cmap='viridis', origin='lower')
    axes[1].set_title(f'Segmented Image ({segmented_image.max()+1} Clusters)', 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(original_image, cmap='gray', origin='lower', alpha=0.7)
    im3 = axes[2].imshow(segmented_image, cmap='viridis', origin='lower', 
                        alpha=0.5, interpolation='nearest')
    axes[2].set_title('Overlay: Original + Segmentation', 
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.show()
    
    # Display model parameters if available
    if model is not None:
        print("\n" + "="*60)
        print("FINAL MODEL PARAMETERS")
        print("="*60)
        for k in range(model.n_components):
            print(f"\nComponent {k}:")
            print(f"  Weight (π_{k}): {model.weights_[k]:.4f}")
            print(f"  Mean (μ_{k}): {model.means_[k]:.4f}")
            print(f"  Scale (α_{k}): {model.alphas_[k]:.4f}")
            print(f"  Shape (β): {model.beta:.2f} (fixed)")
        
        # Plot convergence
        if len(model.log_likelihoods_) > 1:
            plt.figure(figsize=(8, 5))
            plt.plot(model.log_likelihoods_, 'b-', linewidth=2)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Log-Likelihood', fontsize=12)
            plt.title('EM Algorithm Convergence', fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


def apply_ggmm(slice_image, n_components=3, verbose=False):
    """
    Apply GGMM segmentation to a single 2D slice.
    
    This is a wrapper function for segment_image that can be used
    in batch processing of multiple slices.
    
    Parameters:
    -----------
    slice_image : ndarray, shape (height, width)
        2D grayscale image slice
    n_components : int
        Number of segments (clusters)
    verbose : bool
        Whether to print detailed progress information
    
    Returns:
    --------
    segmented_slice : ndarray, shape (height, width)
        Segmented slice with cluster labels
    """
    # Flatten image
    original_shape = slice_image.shape
    flattened = slice_image.flatten()
    
    # Check if slice has sufficient variance
    if np.std(flattened) < 1e-6:
        # Very uniform slice - return single cluster
        if verbose:
            print("  Warning: Slice has very low variance, using single cluster")
        return np.zeros(original_shape, dtype=np.uint8)
    
    # Normalize using MinMaxScaler
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(flattened.reshape(-1, 1)).flatten()
    
    if verbose:
        print(f"  Slice shape: {original_shape}")
        print(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Initialize and fit GGMM
    # Use fewer EM iterations for 3D batch processing to keep runtime reasonable
    model = GeneralizedGaussianMixtureModel(n_components=n_components, 
                                           beta=2.0, max_iter=40, tol=1e-6)
    
    if verbose:
        print("  Fitting GGMM...")
    
    # Suppress warnings and print statements during fitting if not verbose
    import sys
    from io import StringIO
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        old_warnings = warnings.filters[:]
        warnings.filterwarnings('ignore')
    
    try:
        model.fit(normalized, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  Error fitting model: {e}")
        # Fallback: use simple threshold-based segmentation
        thresholds = np.percentile(normalized, np.linspace(0, 100, n_components + 1)[1:-1])
        labels = np.digitize(normalized, thresholds)
        labels = np.clip(labels, 0, n_components - 1)
        segmented_slice = labels.reshape(original_shape)
        return segmented_slice.astype(np.uint8)
    finally:
        if not verbose:
            sys.stdout = old_stdout
            warnings.filters[:] = old_warnings
    
    # Predict cluster assignments
    labels = model.predict(normalized)
    
    # Reshape back to original image shape
    segmented_slice = labels.reshape(original_shape)
    
    return segmented_slice.astype(np.uint8)


def load_3d_volume(data_dir, volume_number, modality_channel=2):
    """
    Load a full 3D MRI volume from individual slice h5 files.
    
    Parameters:
    -----------
    data_dir : str
        Path to data directory
    volume_number : int
        Volume number to load
    modality_channel : int
        Channel index for modality (0=T1, 1=T2, 2=FLAIR, 3=T1CE)
        Default is 2 (FLAIR)
    
    Returns:
    --------
    volume : ndarray, shape (height, width, depth)
        3D MRI volume
    """
    data_path = Path(data_dir)
    
    # Check for data subdirectory
    if (data_path / "data").exists():
        data_path = data_path / "data"
    
    # Find all slice files for this volume
    pattern = f"volume_{volume_number}_slice_*.h5"
    slice_files = sorted(data_path.glob(pattern), 
                        key=lambda x: int(re.search(r'slice_(\d+)', x.name).group(1)))
    
    if not slice_files:
        raise FileNotFoundError(
            f"No slice files found for volume {volume_number}. "
            f"Expected pattern: volume_{volume_number}_slice_*.h5"
        )
    
    print(f"Found {len(slice_files)} slices for volume {volume_number}")
    
    # Load first slice to get dimensions
    with h5py.File(slice_files[0], 'r') as f:
        slice_shape = f['image'].shape[:2]  # (240, 240)
    
    n_slices = len(slice_files)
    height, width = slice_shape
    
    # Initialize empty 3D array
    volume = np.zeros((height, width, n_slices), dtype=np.float32)
    
    # Load each slice
    print("Loading 3D volume...")
    for slice_idx, slice_file in enumerate(slice_files):
        with h5py.File(slice_file, 'r') as f:
            image_data = f['image'][:]
            volume[:, :, slice_idx] = image_data[:, :, modality_channel]
        
        if (slice_idx + 1) % 20 == 0 or slice_idx == 0:
            print(f"  Loaded {slice_idx + 1}/{n_slices} slices...")
    
    print(f"Successfully loaded 3D volume with shape: {volume.shape}")
    print(f"Intensity range: [{volume.min():.4f}, {volume.max():.4f}]")
    
    return volume


def process_3d_volume(volume, n_components=3, verbose_slice=False):
    """
    Process a full 3D volume by applying GGMM to each slice.
    
    Parameters:
    -----------
    volume : ndarray, shape (height, width, depth)
        3D MRI volume
    n_components : int
        Number of segments (clusters) per slice
    verbose_slice : bool
        Whether to print detailed information for each slice
    
    Returns:
    --------
    segmented_volume : ndarray, shape (height, width, depth)
        Segmented 3D volume with cluster labels
    """
    height, width, depth = volume.shape
    
    # Initialize empty 3D array for segmented volume
    segmented_volume = np.zeros((height, width, depth), dtype=np.uint8)
    
    # Suppress sklearn warnings during batch processing
    warnings.filterwarnings('ignore', category=UserWarning, 
                          module='sklearn')
    
    print(f"\nProcessing 3D volume: {volume.shape}")
    print("="*60)
    print(f"Applying GGMM to {depth} slices...")
    print(f"Number of components per slice: {n_components}")
    print("(Warnings suppressed for uniform slices)")
    print("="*60)
    
    # Process each slice along axis=2 (z-axis)
    for slice_idx in range(depth):
        # Extract slice
        slice_image = volume[:, :, slice_idx]
        
        # Apply GGMM segmentation
        if verbose_slice:
            print(f"\nProcessing slice {slice_idx + 1}/{depth}:")
        
        try:
            segmented_slice = apply_ggmm(slice_image, n_components=n_components, 
                                        verbose=verbose_slice)
            segmented_volume[:, :, slice_idx] = segmented_slice
            
            # Progress update
            if (slice_idx + 1) % 10 == 0 or slice_idx == 0 or slice_idx == depth - 1:
                print(f"  Completed slice {slice_idx + 1}/{depth} "
                      f"({100.0 * (slice_idx + 1) / depth:.1f}%)")
        
        except Exception as e:
            print(f"  Error processing slice {slice_idx + 1}: {e}")
            # Use previous slice's segmentation or zeros as fallback
            if slice_idx > 0:
                segmented_volume[:, :, slice_idx] = segmented_volume[:, :, slice_idx - 1]
            else:
                segmented_volume[:, :, slice_idx] = np.zeros((height, width), dtype=np.uint8)
        
        # Memory management: explicitly delete slice data
        del slice_image
        if slice_idx % 50 == 0:
            import gc
            gc.collect()  # Force garbage collection periodically
    
    print(f"\nCompleted processing all {depth} slices!")
    print(f"Segmented volume shape: {segmented_volume.shape}")
    print(f"Label range: [{segmented_volume.min()}, {segmented_volume.max()}]")
    
    return segmented_volume


def save_segmented_volume(segmented_volume, output_path, volume_number):
    """
    Save segmented 3D volume as an h5 file.
    
    Parameters:
    -----------
    segmented_volume : ndarray, shape (height, width, depth)
        Segmented 3D volume
    output_path : str or Path
        Directory where to save the output file
    volume_number : int
        Volume number (for naming the output file)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"volume_{volume_number}_segmented_ggmm.h5"
    
    print(f"\nSaving segmented volume to: {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Save segmented volume
        f.create_dataset('segmentation', data=segmented_volume, 
                        compression='gzip', compression_opts=4)
        
        # Save metadata
        f.attrs['volume_number'] = volume_number
        f.attrs['shape'] = segmented_volume.shape
        f.attrs['dtype'] = str(segmented_volume.dtype)
        f.attrs['n_components'] = int(segmented_volume.max() + 1)
        f.attrs['method'] = 'GGMM'
    
    print(f"Successfully saved segmented volume!")
    print(f"  File: {output_file}")
    print(f"  Shape: {segmented_volume.shape}")
    print(f"  Size: {output_file.stat().st_size / (1024*1024):.2f} MB")


def visualize_3d_segmentation(original_volume, segmented_volume, slice_idx=None):
    """
    Visualize one slice from original and segmented 3D volumes.
    
    Parameters:
    -----------
    original_volume : ndarray, shape (height, width, depth)
        Original 3D MRI volume
    segmented_volume : ndarray, shape (height, width, depth)
        Segmented 3D volume
    slice_idx : int, optional
        Slice index to visualize (default: middle slice)
    """
    if slice_idx is None:
        slice_idx = original_volume.shape[2] // 2
    
    # Extract slices
    original_slice = original_volume[:, :, slice_idx]
    segmented_slice = segmented_volume[:, :, slice_idx]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original slice
    im1 = axes[0].imshow(original_slice, cmap='gray', origin='lower')
    axes[0].set_title(f'Original MRI Slice {slice_idx}\n(FLAIR)', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Segmented slice
    im2 = axes[1].imshow(segmented_slice, cmap='viridis', origin='lower')
    axes[1].set_title(f'Segmented Slice {slice_idx}\n({segmented_slice.max()+1} Clusters)', 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(original_slice, cmap='gray', origin='lower', alpha=0.7)
    im3 = axes[2].imshow(segmented_slice, cmap='viridis', origin='lower', 
                        alpha=0.5, interpolation='nearest')
    axes[2].set_title(f'Overlay: Original + Segmentation\n(Slice {slice_idx})', 
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics for this slice
    print(f"\nSegmentation statistics for slice {slice_idx}:")
    unique_labels, counts = np.unique(segmented_slice, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = (count / segmented_slice.size) * 100
        print(f"  Cluster {label}: {count} pixels ({percentage:.2f}%)")


def main_3d():
    """
    Main function to process a full 3D MRI volume using GGMM.
    """
    print("="*60)
    print("3D MRI VOLUME SEGMENTATION USING GGMM")
    print("="*60)
    
    # Configuration
    data_directory = input("\nEnter the path to the data directory: ").strip()
    data_directory = data_directory.strip('"').strip("'")
    
    volume_number = int(input("Enter volume number (e.g., 1, 2): ").strip())
    
    output_directory = input("Enter output directory for segmented volume "
                            "(press Enter for default: ./segmented_volumes): ").strip()
    if not output_directory:
        output_directory = "./segmented_volumes"
    output_directory = output_directory.strip('"').strip("'")
    
    n_components = int(input("Enter number of clusters (default: 3): ").strip() or "3")
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_directory}")
    print(f"  Volume number: {volume_number}")
    print(f"  Output directory: {output_directory}")
    print(f"  Number of components: {n_components}")
    
    try:
        # Load 3D volume
        print("\n" + "="*60)
        volume = load_3d_volume(data_directory, volume_number, modality_channel=2)
        
        # Process 3D volume
        print("\n" + "="*60)
        segmented_volume = process_3d_volume(volume, n_components=n_components, 
                                            verbose_slice=False)
        
        # Save segmented volume
        print("\n" + "="*60)
        save_segmented_volume(segmented_volume, output_directory, volume_number)
        
        # Visualize one slice
        print("\n" + "="*60)
        print("Visualizing sample slice...")
        visualize_3d_segmentation(volume, segmented_volume, slice_idx=None)
        
        # Print overall statistics
        print("\n" + "="*60)
        print("OVERALL SEGMENTATION STATISTICS")
        print("="*60)
        unique_labels, counts = np.unique(segmented_volume, return_counts=True)
        for label, count in zip(unique_labels, counts):
            percentage = (count / segmented_volume.size) * 100
            print(f"Cluster {label}: {count} voxels ({percentage:.2f}%)")
        
        print("\n" + "="*60)
        print("Processing complete!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function to demonstrate GGMM segmentation on an MRI slice.
    """
    # Configuration
    data_directory = input("Enter the path to the data directory: ").strip()
    data_directory = data_directory.strip('"').strip("'")
    
    volume_number = int(input("Enter volume number (e.g., 1, 2): ").strip())
    slice_number = int(input("Enter slice number (e.g., 50, 75): ").strip())
    
    print(f"\nLoading MRI slice: volume {volume_number}, slice {slice_number}")
    print("="*60)
    
    try:
        # Load MRI slice
        image = load_mri_slice(data_directory, volume_number, slice_number)
        print(f"Loaded slice with shape: {image.shape}")
        print(f"Intensity range: [{image.min():.4f}, {image.max():.4f}]")
        
        # Segment the image
        print("\n" + "="*60)
        segmented_image, model, scaler = segment_image(image, n_components=3)
        
        # Visualize results
        print("\n" + "="*60)
        print("Visualizing results...")
        visualize_segmentation(image, segmented_image, model)
        
        # Print segmentation statistics
        print("\n" + "="*60)
        print("SEGMENTATION STATISTICS")
        print("="*60)
        unique_labels, counts = np.unique(segmented_image, return_counts=True)
        for label, count in zip(unique_labels, counts):
            percentage = (count / segmented_image.size) * 100
            print(f"Cluster {label}: {count} pixels ({percentage:.2f}%)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Choose processing mode
    print("="*60)
    print("GGMM SEGMENTATION - CHOOSE MODE")
    print("="*60)
    print("1. 2D: Process a single slice")
    print("2. 3D: Process a full volume")
    print("="*60)
    
    mode = input("Enter mode (1 or 2, default: 2): ").strip()
    if not mode:
        mode = "2"
    
    if mode == "1":
        main()  # 2D single slice processing
    elif mode == "2":
        main_3d()  # 3D volume processing
    else:
        print(f"Invalid mode: {mode}. Using 3D mode by default.")
        main_3d()

