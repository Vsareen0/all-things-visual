"""
Day 21 Exercise: Multi-Scale Edge Detection with Laplacian of Gaussian
======================================================================

Goal: Build LoG filters at multiple scales, apply them to an image,
find zero crossings, and visualize how sigma controls which edges appear.

Time estimate: 30-60 minutes

Instructions:
1. Run this script as-is to see LoG edge detection at 3 scales
2. Try changing sigma values — what happens at sigma=0.5? sigma=8?
3. Implement the DoG approximation (stretch goal at bottom)
4. Try your own image — notice which edges survive across scales
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data, color


def create_log_kernel(sigma, size=None):
    """
    Create a Laplacian of Gaussian (LoG) kernel.

    The LoG is the Laplacian (second derivative) of a Gaussian.
    It looks like a "Mexican hat" — negative center, positive ring.

    LoG(x,y) = -(1/(pi*sigma^4)) * [1 - (x^2+y^2)/(2*sigma^2)]
               * exp(-(x^2+y^2)/(2*sigma^2))
    """
    if size is None:
        # Kernel should be large enough to capture the Gaussian
        # Rule of thumb: at least 6*sigma across
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1  # Keep it odd for a centered kernel

    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)

    # Gaussian component
    r_squared = x**2 + y**2
    gaussian = np.exp(-r_squared / (2 * sigma**2))

    # LoG formula
    log_kernel = -(1 / (np.pi * sigma**4)) * (1 - r_squared / (2 * sigma**2)) * gaussian

    # Normalize so the kernel sums to zero (important for edge detection!)
    log_kernel -= log_kernel.mean()

    return log_kernel


def find_zero_crossings(log_response, threshold=0.0):
    """
    Find zero crossings in the LoG response.

    An edge exists where the LoG response crosses zero — the sign
    changes between neighboring pixels. We check 4-connected neighbors.

    The threshold filters out weak zero crossings caused by noise.
    """
    rows, cols = log_response.shape
    zero_crossings = np.zeros_like(log_response, dtype=bool)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Check sign changes with each neighbor
            patch = log_response[i - 1:i + 2, j - 1:j + 2]
            p = log_response[i, j]

            # Check if there's a sign change with any neighbor
            if (patch.min() < -threshold and patch.max() > threshold):
                # This pixel is near a zero crossing
                if (p > 0 and patch.min() < -threshold) or \
                   (p < 0 and patch.max() > threshold) or \
                   abs(p) < threshold:
                    zero_crossings[i, j] = True

    return zero_crossings


def main():
    # Load a test image with both fine detail and clear edges
    image = data.camera().astype(float) / 255.0

    # Three scales: fine, medium, coarse
    sigmas = [1, 2, 4]
    colors_for_scales = ['#00d4aa', '#ff6b6b', '#ffd93d']

    fig, axes = plt.subplots(2, len(sigmas) + 1, figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')

    # Show original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', color='white', fontsize=12)
    axes[0, 0].axis('off')

    axes[1, 0].imshow(image, cmap='gray', alpha=0.3)
    axes[1, 0].set_title('All Scales Combined', color='white', fontsize=12)
    axes[1, 0].axis('off')

    for idx, sigma in enumerate(sigmas):
        # Step 1: Create the LoG kernel
        log_kernel = create_log_kernel(sigma)

        # Step 2: Convolve with image
        log_response = ndimage.convolve(image, log_kernel)

        # Step 3: Find zero crossings
        # Threshold scales with sigma (larger scale = stronger edges needed)
        threshold = 0.001 * sigma
        edges = find_zero_crossings(log_response, threshold)

        # Show LoG response (the "Mexican hat" filtered image)
        axes[0, idx + 1].imshow(log_response, cmap='RdBu_r', vmin=-0.05, vmax=0.05)
        axes[0, idx + 1].set_title(f'LoG Response (sigma={sigma})',
                                    color='white', fontsize=12)
        axes[0, idx + 1].axis('off')

        # Show detected edges
        axes[1, idx + 1].imshow(edges, cmap='gray')
        axes[1, idx + 1].set_title(f'Edges (sigma={sigma})',
                                    color='white', fontsize=12)
        axes[1, idx + 1].axis('off')

        # Overlay on combined view
        edge_overlay = np.zeros((*image.shape, 4))
        c = plt.matplotlib.colors.to_rgba(colors_for_scales[idx])
        edge_overlay[edges] = c
        axes[1, 0].imshow(edge_overlay, alpha=0.7)

    # Style all axes
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('#1a1a2e')

    plt.suptitle('Laplacian of Gaussian: Multi-Scale Edge Detection',
                 color='#00d4aa', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('log_multiscale_edges.png', dpi=150, facecolor='#1a1a2e',
                bbox_inches='tight')
    plt.show()
    print("\nSaved: log_multiscale_edges.png")

    # --- Visualize the LoG kernel itself (the "Mexican hat") ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    fig2.patch.set_facecolor('#1a1a2e')

    for idx, sigma in enumerate(sigmas):
        kernel = create_log_kernel(sigma)
        size = kernel.shape[0]
        x = np.arange(size) - size // 2

        # Show center row of the 2D kernel
        center = size // 2
        axes2[idx].plot(x, kernel[center, :], color='#00d4aa', linewidth=2)
        axes2[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes2[idx].fill_between(x, kernel[center, :], 0,
                                 where=kernel[center, :] > 0,
                                 alpha=0.3, color='#00d4aa')
        axes2[idx].fill_between(x, kernel[center, :], 0,
                                 where=kernel[center, :] < 0,
                                 alpha=0.3, color='#ff6b6b')
        axes2[idx].set_title(f'"Mexican Hat" (sigma={sigma})',
                              color='white', fontsize=12)
        axes2[idx].set_facecolor('#1a1a2e')
        axes2[idx].tick_params(colors='white')
        for spine in axes2[idx].spines.values():
            spine.set_color('gray')

    plt.suptitle('LoG Kernel Cross-Sections — The Mexican Hat Shape',
                 color='#00d4aa', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('log_mexican_hat.png', dpi=150, facecolor='#1a1a2e',
                bbox_inches='tight')
    plt.show()
    print("Saved: log_mexican_hat.png")


# ============================================================
# STRETCH GOAL: Difference of Gaussians (DoG) Approximation
# ============================================================
# The DoG approximates the LoG by subtracting two Gaussians
# with slightly different sigmas. SIFT uses k = sqrt(2).
#
# DoG(x,y) = G(x,y,k*sigma) - G(x,y,sigma)
#
# TODO: Implement this function and compare DoG vs LoG results.
#
# def difference_of_gaussians(image, sigma, k=1.6):
#     """Compute DoG: blur at sigma and k*sigma, then subtract."""
#     blur_1 = ndimage.gaussian_filter(image, sigma)
#     blur_2 = ndimage.gaussian_filter(image, k * sigma)
#     return blur_2 - blur_1
#
# Uncomment and add a comparison plot. You should see that
# DoG edges are very similar to LoG edges — that's why SIFT
# uses DoG (it's already computing the blurred images anyway).
# ============================================================


if __name__ == "__main__":
    main()