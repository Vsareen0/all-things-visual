"""
Day 23: Spatial Filtering Toolkit
==================================
Build low-pass, high-pass, band-pass, and Wiener filters from scratch.
Apply them to a noisy synthetic image and compare results.

Run: python day23_spatial_filtering.py
Output: day23_spatial_filtering_results.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal

# ──────────────────────────────────────────────
# 1. Create a synthetic test image (256x256)
# ──────────────────────────────────────────────
def make_test_image(size=256):
    """
    Creates an image with three regions:
    - Left third: sharp vertical edge (step function)
    - Middle third: sinusoidal texture at multiple frequencies
    - Right third: smooth gradient
    """
    img = np.zeros((size, size), dtype=np.float64)
    third = size // 3

    # Left: sharp edge (dark to bright step at the midpoint of the region)
    img[:, :third] = 0.2
    img[:, third // 2 : third] = 0.8

    # Middle: layered sinusoidal texture
    x = np.linspace(0, 1, third)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    # Low-freq + high-freq components
    img[:, third : 2 * third] = (
        0.5
        + 0.2 * np.sin(2 * np.pi * 4 * xx)   # low freq
        + 0.1 * np.sin(2 * np.pi * 20 * yy)  # high freq
    )

    # Right: smooth gradient
    grad = np.linspace(0.1, 0.9, size).reshape(-1, 1)
    img[:, 2 * third :] = grad

    return np.clip(img, 0, 1)


# ──────────────────────────────────────────────
# 2. Add noise
# ──────────────────────────────────────────────
def add_noise(img, sigma=0.05):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 1)


# ──────────────────────────────────────────────
# 3. Filters
# ──────────────────────────────────────────────

def gaussian_kernel(size=5, sigma=1.0):
    """Build a 2D Gaussian kernel from scratch."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def low_pass(img, ksize=5, sigma=1.0):
    """Gaussian blur (low-pass)."""
    kernel = gaussian_kernel(ksize, sigma)
    return ndimage.convolve(img, kernel)


def high_pass(img, ksize=5, sigma=1.0):
    """High-pass = original - low-pass."""
    lp = low_pass(img, ksize, sigma)
    hp = img - lp
    return hp


def band_pass(img, sigma_low=1.0, sigma_high=3.0, ksize=11):
    """Difference of Gaussians (DoG) band-pass filter."""
    g1 = low_pass(img, ksize, sigma_low)
    g2 = low_pass(img, ksize, sigma_high)
    return g1 - g2


def wiener_filter(img, kernel, nsr=0.01):
    """
    Wiener deconvolution in the frequency domain.
    
    img: degraded (blurred + noisy) image
    kernel: the blur kernel (PSF)
    nsr: noise-to-signal power ratio (regularization)
    """
    # Pad kernel to image size
    kernel_pad = np.zeros_like(img)
    kh, kw = kernel.shape
    kernel_pad[:kh, :kw] = kernel
    # Center the kernel (shift so center of kernel is at [0,0])
    kernel_pad = np.roll(kernel_pad, -(kh // 2), axis=0)
    kernel_pad = np.roll(kernel_pad, -(kw // 2), axis=1)

    # FFT
    IMG = np.fft.fft2(img)
    H = np.fft.fft2(kernel_pad)

    # Wiener formula: H* / (|H|^2 + NSR)
    H_conj = np.conj(H)
    H_mag_sq = np.abs(H) ** 2
    W = H_conj / (H_mag_sq + nsr)

    restored = np.real(np.fft.ifft2(IMG * W))
    return np.clip(restored, 0, 1)


# ──────────────────────────────────────────────
# 4. Main: Run everything and visualize
# ──────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # Build the test image and noisy version
    clean = make_test_image()
    noisy = add_noise(clean, sigma=0.05)

    # Apply filters to the noisy image
    lp_result = low_pass(noisy, ksize=7, sigma=1.5)
    hp_result = high_pass(noisy, ksize=7, sigma=1.5)
    bp_result = band_pass(noisy, sigma_low=1.0, sigma_high=3.0, ksize=11)

    # For Wiener: simulate motion blur then try to undo it
    blur_kernel = gaussian_kernel(7, 2.0)
    blurred_noisy = ndimage.convolve(noisy, blur_kernel)
    wiener_result = wiener_filter(blurred_noisy, blur_kernel, nsr=0.02)

    # ── Plotting ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    titles_top = ["Clean Original", "Noisy (σ=0.05)", "Low-Pass (Gaussian)", "High-Pass"]
    imgs_top = [clean, noisy, lp_result, hp_result]

    titles_bot = ["Band-Pass (DoG)", "Blurred + Noisy", "Wiener Restored", "LP + HP = Original?"]
    # Verify complementarity: LP + HP should ≈ original
    reconstructed = lp_result + hp_result
    imgs_bot = [bp_result, blurred_noisy, wiener_result, reconstructed]

    for ax, title, img in zip(axes[0], titles_top, imgs_top):
        if "High" in title or "Band" in title:
            # Center around zero for high-pass / band-pass display
            vmax = max(abs(img.min()), abs(img.max()))
            ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    for ax, title, img in zip(axes[1], titles_bot, imgs_bot):
        if "Band" in title:
            vmax = max(abs(img.min()), abs(img.max()))
            ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Day 23: Spatial Filtering — Four Flavors", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("day23_spatial_filtering_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: day23_spatial_filtering_results.png")

    # ── Reconstruction error ──
    error = np.max(np.abs(reconstructed - noisy))
    print(f"Max reconstruction error (LP + HP vs noisy): {error:.2e}")
    print("(Should be near machine epsilon — confirms complementarity!)")

    # ──────────────────────────────────────────────
    # 5. CHALLENGE: Wiener NSR sweep
    # ──────────────────────────────────────────────
    print("\n── Challenge: Wiener NSR Sweep ──")
    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))
    nsr_values = [0.0001, 0.001, 0.01, 0.1, 1.0]

    for ax, nsr in zip(axes2, nsr_values):
        result = wiener_filter(blurred_noisy, blur_kernel, nsr=nsr)
        mse = np.mean((result - clean) ** 2)
        ax.imshow(result, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"NSR={nsr}\nMSE={mse:.4f}", fontsize=10)
        ax.axis("off")

    plt.suptitle("Wiener Filter: NSR Tradeoff (lower NSR = more aggressive restoration)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig("day23_wiener_nsr_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: day23_wiener_nsr_sweep.png")
    print("Notice: too-low NSR amplifies noise; too-high NSR over-smooths. The sweet spot is in between.")