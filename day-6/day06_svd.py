"""
Day 6 Exercise: SVD Image Compression
======================================
Explore how Singular Value Decomposition can compress images by
keeping only the most important patterns.

Requirements: pip install numpy matplotlib Pillow requests
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import os

# ── Step 1: Load an image ──────────────────────────────────────────
# We'll use a sample image. Replace IMAGE_PATH with your own file if you like.
IMAGE_PATH = None  # Set to a local file path like "my_photo.jpg" to use your own

def load_image():
    """Load a sample image or a user-provided one."""
    if IMAGE_PATH and os.path.exists(IMAGE_PATH):
        print(f"Loading image from: {IMAGE_PATH}")
        img = Image.open(IMAGE_PATH)
    else:
        print("Downloading sample image (astronaut from skimage)...")
        # Use a public-domain test image
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/640px-Camponotus_flavomarginatus_ant.jpg"
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
        except Exception:
            # Fallback: generate a synthetic test image with gradients and shapes
            print("Download failed — generating a synthetic test image instead.")
            img = generate_test_image()

    # Resize to manageable size for SVD speed
    img = img.resize((400, 300))
    return np.array(img, dtype=np.float64) / 255.0  # Normalize to [0, 1]


def generate_test_image():
    """Create a synthetic image with gradients and shapes for testing."""
    w, h = 400, 300
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background
    for y in range(h):
        for x in range(w):
            img[y, x] = [
                int(255 * x / w),       # Red gradient left-right
                int(255 * y / h),       # Green gradient top-bottom
                int(128 + 60 * np.sin(x / 20.0))  # Blue sinusoidal
            ]

    # Add a bright circle
    cy, cx, r = 150, 200, 60
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
    img[mask] = [255, 220, 100]

    return Image.fromarray(img)


# ── Step 2: SVD on each color channel ──────────────────────────────

def svd_compress_channel(channel, k):
    """
    Compress a single 2D channel using rank-k SVD approximation.

    Parameters:
        channel: 2D numpy array (m x n)
        k: number of singular values to keep

    Returns:
        Reconstructed channel (m x n), clipped to [0, 1]
    """
    U, s, Vt = np.linalg.svd(channel, full_matrices=False)
    # Keep only top-k components
    reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return np.clip(reconstructed, 0, 1)


def svd_compress_image(image, k):
    """Compress an RGB image using rank-k SVD on each channel."""
    compressed = np.zeros_like(image)
    for c in range(3):  # R, G, B
        compressed[:, :, c] = svd_compress_channel(image[:, :, c], k)
    return compressed


# ── Step 3: Compression ratio calculation ──────────────────────────

def compression_ratio(m, n, k):
    """
    Calculate the compression ratio for rank-k SVD.

    Original storage: m * n
    SVD storage:      k * (m + n + 1)  [U columns + Vt rows + singular values]
    """
    original = m * n
    compressed = k * (m + 1 + n)  # k columns of U, k singular values, k rows of Vt
    return original / compressed


# ── Step 4: Run the experiment ─────────────────────────────────────

def main():
    # Load image
    image = load_image()
    m, n = image.shape[0], image.shape[1]
    max_rank = min(m, n)

    print(f"Image size: {m} x {n} pixels")
    print(f"Maximum possible rank: {max_rank}")
    print()

    # Values of k to try
    k_values = [1, 5, 10, 20, 50, 100, 200]
    k_values = [k for k in k_values if k <= max_rank]  # Filter valid k values

    # ── Compute SVD once, reuse for all k values ───────────────────
    print("Computing SVD (this may take a few seconds)...")
    channels_svd = []
    for c in range(3):
        U, s, Vt = np.linalg.svd(image[:, :, c], full_matrices=False)
        channels_svd.append((U, s, Vt))
    print("Done!\n")

    # ── Print singular value magnitudes ────────────────────────────
    print("First 20 singular values (Red channel):")
    s_red = channels_svd[0][1]
    for i in range(min(20, len(s_red))):
        bar = "█" * int(s_red[i] / s_red[0] * 40)
        print(f"  σ_{i+1:3d} = {s_red[i]:8.2f}  {bar}")
    print(f"  ...({len(s_red)} total)\n")

    # ── Reconstruct for each k ─────────────────────────────────────
    reconstructions = {}
    for k in k_values:
        compressed = np.zeros_like(image)
        for c in range(3):
            U, s, Vt = channels_svd[c]
            compressed[:, :, c] = np.clip(
                U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :], 0, 1
            )
        ratio = compression_ratio(m, n, k)
        reconstructions[k] = compressed
        print(f"k = {k:4d}  |  Compression ratio: {ratio:6.1f}x  |  Storage: {k*(m+1+n)*3:,} numbers (vs {m*n*3:,} original)")

    # ── Plot 1: Reconstructions side by side ───────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("SVD Image Compression: How Many Patterns Do You Need?", fontsize=14)

    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Original\n({m*n*3:,} values)")
    axes[0, 0].axis("off")

    # Compressed versions
    for idx, k in enumerate(k_values):
        row, col = divmod(idx + 1, 4)
        ratio = compression_ratio(m, n, k)
        axes[row, col].imshow(reconstructions[k])
        axes[row, col].set_title(f"k = {k}\n({ratio:.1f}x compression)")
        axes[row, col].axis("off")

    # Hide any unused subplot
    total_plots = len(k_values) + 1
    for idx in range(total_plots, 8):
        row, col = divmod(idx, 4)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("day06_svd_reconstructions.png", dpi=150, bbox_inches="tight")
    print("\nSaved: day06_svd_reconstructions.png")
    plt.show()

    # ── Plot 2: Singular value decay ───────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Singular Value Analysis", fontsize=14)

    colors = ["red", "green", "blue"]
    labels = ["Red", "Green", "Blue"]
    for c in range(3):
        s = channels_svd[c][1]
        ax1.plot(s, color=colors[c], alpha=0.8, label=labels[c])
        ax1.axvline(x=50, color="gray", linestyle="--", alpha=0.5)

    ax1.set_xlabel("Index (rank)")
    ax1.set_ylabel("Singular value magnitude")
    ax1.set_title("Singular Values (linear scale)")
    ax1.legend()
    ax1.annotate("k=50 (sweet spot?)", xy=(50, 0), fontsize=9, color="gray")

    # Cumulative energy (what fraction of total "information" is captured?)
    for c in range(3):
        s = channels_svd[c][1]
        cumulative = np.cumsum(s ** 2) / np.sum(s ** 2) * 100
        ax2.plot(cumulative, color=colors[c], alpha=0.8, label=labels[c])

    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(y=99, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Number of singular values kept (k)")
    ax2.set_ylabel("% of total energy captured")
    ax2.set_title("Cumulative Energy")
    ax2.legend()
    ax2.annotate("95%", xy=(0, 95.5), fontsize=9, color="gray")
    ax2.annotate("99%", xy=(0, 99.5), fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig("day06_svd_singular_values.png", dpi=150, bbox_inches="tight")
    print("Saved: day06_svd_singular_values.png")
    plt.show()

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Most singular values are tiny — images are highly redundant.")
    print("2. The 'elbow' in the singular value curve shows where")
    print("   important structure ends and noise/fine detail begins.")
    print("3. SVD isn't used directly in JPEG/H.264 (they use DCT),")
    print("   but the PRINCIPLE is the same: find the most important")
    print("   patterns and throw away the rest.")
    print("4. Try the stretch challenge: compare a photo vs. text")
    print("   screenshot — text needs many more singular values")
    print("   because it has lots of sharp, unpredictable edges.")
    print("=" * 60)


if __name__ == "__main__":
    main()
