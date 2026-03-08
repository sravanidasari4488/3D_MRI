"""
Generalized Gaussian Probability Density Function Implementation

The Generalized Gaussian Distribution (GGD) is a flexible family of distributions
that includes the normal distribution as a special case. It allows for different
tail behaviors and kurtosis through the shape parameter beta.

PDF Formula:
    f(x; μ, α, β) = (β / (2α * Γ(1/β))) * exp(-(|x - μ|/α)^β)

Where:
    μ (mu):    Location parameter (mean/median/mode)
    α (alpha): Scale parameter (controls spread, similar to standard deviation)
    β (beta):  Shape parameter (controls tail behavior and kurtosis)
    Γ:         Gamma function
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def generalized_gaussian_pdf(x, mu, alpha, beta):
    """
    Compute the Generalized Gaussian Probability Density Function.
    
    Parameters:
    -----------
    x : array_like
        Input values at which to evaluate the PDF
    mu : float
        Location parameter (mean/median/mode of the distribution)
    alpha : float
        Scale parameter (controls the spread of the distribution)
        - Larger alpha = wider distribution
        - Similar role to standard deviation in normal distribution
    beta : float
        Shape parameter (controls tail behavior and kurtosis)
        - beta = 1: Laplace distribution (heavy tails)
        - beta = 2: Normal/Gaussian distribution (standard normal)
        - beta > 2: Sub-Gaussian (lighter tails, more peaked)
        - beta < 1: Super-Gaussian (heavier tails, flatter)
    
    Returns:
    --------
    pdf : ndarray
        Probability density values at x
    
    Notes:
    ------
    - For numerical stability, we compute the exponent carefully
    - The normalization constant ensures the PDF integrates to 1
    - When beta = 2, this reduces to the standard normal distribution
      with standard deviation = alpha / sqrt(2)
    """
    # Convert inputs to numpy arrays for vectorized operations
    x = np.asarray(x, dtype=np.float64)
    mu = float(mu)
    alpha = float(alpha)
    beta = float(beta)
    
    # Ensure alpha and beta are positive (required for valid PDF)
    if alpha <= 0:
        raise ValueError("alpha (scale parameter) must be positive")
    if beta <= 0:
        raise ValueError("beta (shape parameter) must be positive")
    
    # Compute the absolute deviation from the location parameter
    abs_dev = np.abs(x - mu)
    
    # Normalize by the scale parameter
    normalized = abs_dev / alpha
    
    # Compute the exponent: -(normalized)^beta
    # Use np.clip to prevent overflow for very large values
    exponent = np.power(normalized, beta)
    exponent = np.clip(exponent, 0, 700)  # Prevent overflow in exp
    exp_term = np.exp(-exponent)
    
    # Compute the normalization constant
    # C = β / (2α * Γ(1/β))
    # This ensures the PDF integrates to 1 over the entire real line
    normalization_constant = beta / (2.0 * alpha * gamma(1.0 / beta))
    
    # Compute the PDF
    pdf = normalization_constant * exp_term
    
    # Handle any potential numerical issues (NaN, Inf)
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    
    return pdf


def plot_generalized_gaussian_comparison():
    """
    Plot Generalized Gaussian PDFs for different beta values.
    
    This visualization demonstrates:
    - How beta controls the shape of the distribution
    - The relationship between beta and tail behavior
    - Comparison with the standard normal distribution (beta=2)
    """
    # Define parameters
    mu = 0.0      # Location parameter (centered at zero)
    alpha = 1.0   # Scale parameter (fixed for comparison)
    
    # Beta values to plot
    # beta = 1: Laplace distribution (heavier tails than normal)
    # beta = 2: Normal distribution (standard case)
    # beta = 4: Sub-Gaussian (lighter tails, more peaked)
    beta_values = [1, 2, 4]
    
    # Create x-axis values
    # Use a range that shows the tail behavior
    x = np.linspace(-5, 5, 1000)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Linear scale (shows overall shape)
    for beta in beta_values:
        pdf = generalized_gaussian_pdf(x, mu, alpha, beta)
        label = f'β = {beta}'
        if beta == 2:
            label += ' (Normal/Gaussian)'
        elif beta == 1:
            label += ' (Laplace)'
        ax1.plot(x, pdf, label=label, linewidth=2)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Probability Density f(x)', fontsize=12)
    ax1.set_title('Generalized Gaussian PDF: Effect of Shape Parameter β\n(α = 1.0, μ = 0.0)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    
    # Plot 2: Log scale (better shows tail behavior)
    for beta in beta_values:
        pdf = generalized_gaussian_pdf(x, mu, alpha, beta)
        # Use log scale, but avoid log(0)
        pdf_log = np.log10(pdf + 1e-10)
        label = f'β = {beta}'
        if beta == 2:
            label += ' (Normal/Gaussian)'
        elif beta == 1:
            label += ' (Laplace)'
        ax2.plot(x, pdf_log, label=label, linewidth=2)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('log₁₀(Probability Density)', fontsize=12)
    ax2.set_title('Generalized Gaussian PDF: Tail Behavior (Log Scale)\n(α = 1.0, μ = 0.0)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 5)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary information
    print("\n" + "="*70)
    print("GENERALIZED GAUSSIAN DISTRIBUTION - PARAMETER ROLES")
    print("="*70)
    print("\n1. LOCATION PARAMETER (μ, mu):")
    print("   - Controls where the distribution is centered")
    print("   - Acts as the mean, median, and mode")
    print("   - Shifting μ moves the entire distribution left or right")
    
    print("\n2. SCALE PARAMETER (α, alpha):")
    print("   - Controls the spread/width of the distribution")
    print("   - Larger α = wider distribution (more spread out)")
    print("   - Smaller α = narrower distribution (more concentrated)")
    print("   - Similar role to standard deviation in normal distribution")
    print("   - For β = 2 (normal case): σ = α / √2")
    
    print("\n3. SHAPE PARAMETER (β, beta):")
    print("   - Controls tail behavior and kurtosis (peakedness)")
    print("   - β = 1: Laplace distribution")
    print("     * Heavy tails (more probability in tails)")
    print("     * Higher kurtosis than normal")
    print("   - β = 2: Normal/Gaussian distribution")
    print("     * Standard bell curve")
    print("     * Exponential decay in tails: exp(-x²)")
    print("   - β > 2: Sub-Gaussian")
    print("     * Lighter tails (less probability in tails)")
    print("     * More peaked than normal")
    print("     * Faster decay: exp(-x^β) with β > 2")
    print("   - β < 1: Super-Gaussian")
    print("     * Very heavy tails")
    print("     * Flatter than normal")
    
    print("\n4. DIFFERENCE FROM NORMAL DISTRIBUTION:")
    print("   - Normal distribution is a special case (β = 2)")
    print("   - GGD allows modeling of non-normal tail behavior")
    print("   - Can capture distributions with:")
    print("     * Heavier tails (β < 2): more outliers expected")
    print("     * Lighter tails (β > 2): fewer outliers expected")
    print("   - Useful in signal processing, image processing, and")
    print("     modeling data with non-Gaussian characteristics")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run the visualization
    plot_generalized_gaussian_comparison()
    
    # Example: Compute PDF values for specific parameters
    print("\nExample: Computing PDF values")
    print("-" * 50)
    x_example = np.array([-2, -1, 0, 1, 2])
    mu_example = 0.0
    alpha_example = 1.0
    beta_example = 2.0  # Normal distribution
    
    pdf_values = generalized_gaussian_pdf(x_example, mu_example, alpha_example, beta_example)
    print(f"Parameters: μ = {mu_example}, α = {alpha_example}, β = {beta_example}")
    print(f"x values: {x_example}")
    print(f"PDF values: {pdf_values}")
    print(f"\nNote: For β = 2, this should match the standard normal distribution")
    print(f"      (with σ = α/√2 = {alpha_example/np.sqrt(2):.4f})")

