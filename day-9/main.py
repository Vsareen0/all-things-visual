"""
Day 9 Exercise: Build Your Own 2D Convolution Engine
=====================================================
Implement convolution from scratch, apply different kernels to an image,
and compare results visually.

Requirements: pip install numpy pillow matplotlib scipy
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform 2D convolution on a grayscale image with a given kernel.
    
    Steps:
    1. Flip the kernel (180-degree rotation) — this is what makes it
       convolution rather than correlation.
    2. Pad the image with zeros so the output is the same size as input.
    3. Slide the kernel across every pixel, computing the weighted sum.
    
    Args:
        image: 2D numpy array (grayscale image), values 0-255 or 0.0-1.0
        kernel: 2D numpy array (e.g., 3x3 or 5x5)
    
    Returns:
        2D numpy array — the convolved image (same size as input)
    """
    # Step 1: Flip the kernel (convolution = correlation with flipped kernel)
    kernel_flipped = kernel[::-1, ::-1]
    
    kh, kw = kernel_flipped.shape
    ih, iw = image.shape
    
    # Step 2: Zero-pad the image
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.zeros((ih + 2 * pad_h, iw + 2 * pad_w))
    padded[pad_h:pad_h + ih, pad_w:pad_w + iw] = image
    
    # Step 3: Slide and compute
    output = np.zeros_like(image, dtype=np.float64)
    
    for i in range(ih):
        for j in range(iw):
            # Extract the region under the kernel
            region = padded[i:i + kh, j:j + kw]
            # Element-wise multiply and sum
            output[i, j] = np.sum(region * kernel_flipped)
    
    return output


def main():
    # --- Load or generate a test image ---
    # Try loading a real image; fall back to a synthetic one
    try:
        img = Image.open("test_image.jpg").convert("L")  # grayscale
        image = np.array(img, dtype=np.float64)
    except FileNotFoundError:
        print("No test_image.jpg found — generating a synthetic image.")
        print("(For better results, place any .jpg image named 'test_image.jpg' here.)\n")
        # Create a synthetic image: gradient with some shapes
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        xx, yy = np.meshgrid(x, y)
        image = np.zeros((256, 256))
        # Add a gradient
        image += xx * 128
        # Add a circle
        circle = ((xx - 0.5)**2 + (yy - 0.5)**2) < 0.04
        image[circle] = 255
        # Add a rectangle
        image[50:80, 150:220] = 200
        # Add some noise
        image += np.random.normal(0, 10, image.shape)
        image = np.clip(image, 0, 255)

    # --- Define kernels ---
    
    # 1. Box blur (5x5) — averages all neighbors equally
    #    Every pixel becomes the mean of its 25 nearest neighbors.
    #    This kills high-frequency detail (edges, noise).
    box_blur_5x5 = np.ones((5, 5)) / 25.0
    
    # 2. Sharpening kernel (3x3)
    #    Center pixel gets boosted (+5), neighbors get subtracted (-1).
    #    This amplifies the difference between a pixel and its surroundings,
    #    making edges and fine detail more prominent.
    sharpen_3x3 = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64)
    
    # 3. Sobel edge detector (horizontal edges)
    #    Detects vertical gradients — responds where brightness changes
    #    from top to bottom. The top row is positive, bottom negative.
    sobel_horizontal = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float64)

    # --- Apply convolution ---
    print("Applying box blur (5x5)...")
    t0 = time.time()
    blurred = convolve2d(image, box_blur_5x5)
    t_blur = time.time() - t0
    print(f"  Done in {t_blur:.2f}s")

    print("Applying sharpening kernel...")
    t0 = time.time()
    sharpened = convolve2d(image, sharpen_3x3)
    t_sharp = time.time() - t0
    print(f"  Done in {t_sharp:.2f}s")

    print("Applying Sobel edge detector...")
    t0 = time.time()
    edges = convolve2d(image, sobel_horizontal)
    t_edge = time.time() - t0
    print(f"  Done in {t_edge:.2f}s")

    # --- Bonus: Compare with scipy ---
    try:
        from scipy.signal import convolve2d as scipy_conv2d
        print("\nBonus: Comparing with scipy.signal.convolve2d...")
        t0 = time.time()
        blurred_scipy = scipy_conv2d(image, box_blur_5x5, mode='same', boundary='fill')
        t_scipy = time.time() - t0
        print(f"  scipy blur: {t_scipy:.4f}s vs yours: {t_blur:.2f}s")
        print(f"  Speedup: {t_blur / t_scipy:.0f}x faster")
        
        # Verify correctness: your result should be very close to scipy's
        max_diff = np.max(np.abs(blurred - blurred_scipy))
        print(f"  Max pixel difference: {max_diff:.6f} (should be near zero)")
    except ImportError:
        print("\n(Install scipy to compare performance: pip install scipy)")

    # --- Display results ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.clip(blurred, 0, 255), cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title("Box Blur 5x5")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(np.clip(sharpened, 0, 255), cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title("Sharpened")
    axes[1, 0].axis('off')
    
    # For edges, use absolute value and auto-scale (edges can be negative)
    axes[1, 1].imshow(np.abs(edges), cmap='gray')
    axes[1, 1].set_title("Sobel Edges (horizontal)")
    axes[1, 1].axis('off')
    
    plt.suptitle("Day 9: Convolution with Different Kernels", fontsize=14)
    plt.tight_layout()
    plt.savefig("day09_convolution_results.png", dpi=150)
    plt.show()
    print("\nResults saved to day09_convolution_results.png")


if __name__ == "__main__":
    main()