"""
Day 8: Bayer Demosaicing — Build Your Own Demosaicing Pipeline
==============================================================
Goal: Simulate a Bayer sensor mosaic, then reconstruct full color using
      (1) bilinear interpolation and (2) edge-directed interpolation.
      Compare both to the original image.

Requirements: pip install numpy matplotlib scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics, io
from scipy.ndimage import convolve


# ── Step 1: Load a test image ──────────────────────────────────────────────
# Using skimage's built-in astronaut image (512x512 RGB)
# Feel free to replace with your own: img = io.imread("your_image.png")
img = data.astronaut().astype(np.float64) / 255.0
H, W, _ = img.shape
print(f"Image size: {W}x{H}")


# ── Step 2: Simulate Bayer mosaic (GRBG pattern) ──────────────────────────
# The Bayer pattern repeats every 2x2 block:
#   Row 0: G R G R G R ...
#   Row 1: B G B G B G ...
#   Row 2: G R G R G R ...
#   ...
#
# At each pixel, we keep ONLY the one channel that the Bayer filter allows
# through, and set the other two to zero (unknown).

def create_bayer_mosaic(image):
    """Simulate a Bayer sensor by keeping only one channel per pixel."""
    h, w, _ = image.shape
    mosaic = np.zeros((h, w), dtype=np.float64)
    # Also create a mask telling us which channel each pixel has
    # 0 = Red, 1 = Green, 2 = Blue
    pattern = np.zeros((h, w), dtype=np.int32)

    for y in range(h):
        for x in range(w):
            if y % 2 == 0 and x % 2 == 0:      # Green (top-left of 2x2)
                mosaic[y, x] = image[y, x, 1]
                pattern[y, x] = 1
            elif y % 2 == 0 and x % 2 == 1:     # Red
                mosaic[y, x] = image[y, x, 0]
                pattern[y, x] = 0
            elif y % 2 == 1 and x % 2 == 0:     # Blue
                mosaic[y, x] = image[y, x, 2]
                pattern[y, x] = 2
            else:                                  # Green (bottom-right)
                mosaic[y, x] = image[y, x, 1]
                pattern[y, x] = 1

    return mosaic, pattern


print("Creating Bayer mosaic...")
mosaic, pattern = create_bayer_mosaic(img)


# ── Step 3: Bilinear interpolation demosaicing ────────────────────────────
# For each missing channel at each pixel, average the nearest neighbors
# that have that channel's value.
#
# We can do this efficiently with convolution kernels:

def demosaic_bilinear(mosaic, pattern):
    """Reconstruct full color using bilinear interpolation."""
    h, w = mosaic.shape
    result = np.zeros((h, w, 3), dtype=np.float64)

    # Create separate channel planes from the mosaic
    r_plane = np.where(pattern == 0, mosaic, 0)
    g_plane = np.where(pattern == 1, mosaic, 0)
    b_plane = np.where(pattern == 2, mosaic, 0)

    # Kernel for averaging green values (green appears in a checkerboard)
    # At a non-green pixel, the 4 direct neighbors are green
    green_kernel = np.array([
        [0, 1, 0],
        [1, 4, 1],
        [0, 1, 0]
    ], dtype=np.float64) / 4.0

    # Kernel for averaging red/blue at same-color positions (they appear
    # on a grid spaced 2 apart, so we need a 3x3 kernel hitting corners)
    rb_cross_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64) / 4.0

    # For red and blue, we need different handling.
    # Simple approach: convolve each plane with an averaging kernel,
    # then divide by a "count" kernel to normalize.
    r_mask = (pattern == 0).astype(np.float64)
    g_mask = (pattern == 1).astype(np.float64)
    b_mask = (pattern == 2).astype(np.float64)

    avg_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64)

    # Interpolate each channel
    for ch_idx, (plane, mask) in enumerate([(r_plane, r_mask),
                                              (g_plane, g_mask),
                                              (b_plane, b_mask)]):
        value_sum = convolve(plane, avg_kernel, mode='reflect')
        count_sum = convolve(mask, avg_kernel, mode='reflect')
        count_sum = np.maximum(count_sum, 1e-10)  # avoid division by zero
        result[:, :, ch_idx] = value_sum / count_sum

    return np.clip(result, 0, 1)


print("Demosaicing with bilinear interpolation...")
bilinear_result = demosaic_bilinear(mosaic, pattern)


# ── Step 4: Edge-directed interpolation ───────────────────────────────────
# Before averaging, check if there's a strong edge in the horizontal or
# vertical direction. If so, only average along the direction with the
# SMALLER gradient (i.e., the smooth direction).

def demosaic_edge_directed(mosaic, pattern):
    """Reconstruct full color using simple edge-directed interpolation."""
    h, w = mosaic.shape
    result = np.zeros((h, w, 3), dtype=np.float64)

    # Start with bilinear as a base for green channel
    g_plane = np.where(pattern == 1, mosaic, 0)
    g_mask = (pattern == 1).astype(np.float64)

    avg_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
    g_sum = convolve(g_plane, avg_kernel, mode='reflect')
    g_count = convolve(g_mask, avg_kernel, mode='reflect')
    g_count = np.maximum(g_count, 1e-10)
    green_bilinear = g_sum / g_count

    # Now do edge-directed green interpolation at non-green pixels
    green_ed = green_bilinear.copy()
    # At red and blue pixels, we have green neighbors N/S/E/W
    # Check horizontal vs vertical gradient in the mosaic
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            if pattern[y, x] != 1:  # non-green pixel
                # Horizontal gradient (using mosaic values)
                grad_h = abs(mosaic[y, x - 1] - mosaic[y, x + 1])
                # Vertical gradient
                grad_v = abs(mosaic[y - 1, x] - mosaic[y + 1, x])

                if grad_h < grad_v:
                    # Smooth horizontally → interpolate along horizontal
                    # Average left and right green neighbors
                    vals = []
                    if pattern[y, x - 1] == 1:
                        vals.append(mosaic[y, x - 1])
                    if pattern[y, x + 1] == 1:
                        vals.append(mosaic[y, x + 1])
                    if vals:
                        green_ed[y, x] = np.mean(vals)
                elif grad_v < grad_h:
                    # Smooth vertically → interpolate along vertical
                    vals = []
                    if pattern[y - 1, x] == 1:
                        vals.append(mosaic[y - 1, x])
                    if pattern[y + 1, x] == 1:
                        vals.append(mosaic[y + 1, x])
                    if vals:
                        green_ed[y, x] = np.mean(vals)
                # If equal, keep bilinear result

    result[:, :, 1] = green_ed

    # For red and blue, use guided interpolation:
    # Reconstruct R and B using the green channel as a guide.
    # At a green pixel, estimate R as: R_neighbor + (G_here - G_neighbor)
    r_plane = np.where(pattern == 0, mosaic, 0)
    b_plane = np.where(pattern == 2, mosaic, 0)
    r_mask = (pattern == 0).astype(np.float64)
    b_mask = (pattern == 2).astype(np.float64)

    # Use color-difference approach: interpolate (R-G) and (B-G) instead
    # of raw R and B. This leverages interchannel correlation.
    rg_diff = np.where(pattern == 0, mosaic - green_ed, 0)
    bg_diff = np.where(pattern == 2, mosaic - green_ed, 0)

    rg_sum = convolve(rg_diff, avg_kernel, mode='reflect')
    rg_count = convolve(r_mask, avg_kernel, mode='reflect')
    rg_count = np.maximum(rg_count, 1e-10)
    result[:, :, 0] = green_ed + rg_sum / rg_count

    bg_sum = convolve(bg_diff, avg_kernel, mode='reflect')
    bg_count = convolve(b_mask, avg_kernel, mode='reflect')
    bg_count = np.maximum(bg_count, 1e-10)
    result[:, :, 2] = green_ed + bg_sum / bg_count

    return np.clip(result, 0, 1)


print("Demosaicing with edge-directed interpolation (this takes a moment)...")
edge_result = demosaic_edge_directed(mosaic, pattern)


# ── Step 5: Compare results ──────────────────────────────────────────────
psnr_bilinear = metrics.peak_signal_noise_ratio(img, bilinear_result)
psnr_edge = metrics.peak_signal_noise_ratio(img, edge_result)

print(f"\nResults:")
print(f"  Bilinear PSNR:      {psnr_bilinear:.2f} dB")
print(f"  Edge-directed PSNR: {psnr_edge:.2f} dB")
print(f"  Improvement:        {psnr_edge - psnr_bilinear:+.2f} dB")


# ── Visualization ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Top row: full images
axes[0, 0].imshow(img)
axes[0, 0].set_title("Original")
axes[0, 1].imshow(bilinear_result)
axes[0, 1].set_title(f"Bilinear ({psnr_bilinear:.1f} dB)")
axes[0, 2].imshow(edge_result)
axes[0, 2].set_title(f"Edge-Directed ({psnr_edge:.1f} dB)")

# Bottom row: zoomed-in crop (look for demosaicing artifacts)
# Crop a region with edges — the astronaut's face/helmet area
cy, cx, cs = 80, 200, 80  # center y, center x, crop size
crop = slice(cy, cy + cs), slice(cx, cx + cs)

axes[1, 0].imshow(img[crop])
axes[1, 0].set_title("Original (zoomed)")
axes[1, 1].imshow(bilinear_result[crop])
axes[1, 1].set_title("Bilinear (zoomed)")
axes[1, 2].imshow(edge_result[crop])
axes[1, 2].set_title("Edge-Directed (zoomed)")

for ax in axes.flat:
    ax.axis('off')

plt.suptitle("Day 8: Bayer Demosaicing Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("day08_demosaic_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved comparison to day08_demosaic_comparison.png")
print("\n── Exercises to try ──")
print("1. Zoom into different regions — where do you see color fringing?")
print("2. Try loading a photo with fine textures (fabric, brick, grass)")
print("3. Compute per-channel PSNR — which channel benefits most from edge-direction?")
print("4. [Challenge] Implement a 5x5 kernel for the edge-directed method")