"""
Day 34 — Why PSNR lies and perceptual metrics don't.

We build TWO degradations of the same image tuned to nearly equal PSNR:
  (a) mild uniform noise everywhere
  (b) strong Gaussian blur
PSNR rates them about the same. SSIM (structure-aware) does not -- and your
own eyes won't either. That gap is exactly why Netflix built VMAF: a single
pixel-error number is blind to *what humans actually notice*.

Run:
    pip install numpy scipy scikit-image matplotlib --break-system-packages
    python day34_perceptual_quality.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def make_scene(n=256):
    """Smooth sky over busy texture: noise hides in texture, blur kills detail."""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    sky = 70 + 110 * (y / n)                          # smooth gradient
    tex = 128 + 60 * np.sin(x / 2.5) * np.cos(y / 2.0)  # high-frequency detail
    img = np.where(y > 0.55 * n, tex, sky)
    img[110:180, 80:150] = 25                          # a hard-edged object
    return np.clip(img, 0, 255)


def match_psnr_noise(img, target_db, lo=1.0, hi=80.0):
    """Binary-search noise sigma so noisy image hits target PSNR."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(img.shape)
    for _ in range(40):
        mid = (lo + hi) / 2
        noisy = np.clip(img + mid * base, 0, 255)
        p = psnr(img, noisy)
        if p > target_db:   # too clean -> add more noise
            lo = mid
        else:
            hi = mid
    return np.clip(img + ((lo + hi) / 2) * base, 0, 255)


def match_psnr_blur(img, target_db, lo=0.3, hi=8.0):
    """Binary-search blur sigma so blurred image hits target PSNR."""
    for _ in range(40):
        mid = (lo + hi) / 2
        blurred = gaussian_filter(img, mid)
        p = psnr(img, blurred)
        if p > target_db:   # too sharp -> blur more
            lo = mid
        else:
            hi = mid
    return gaussian_filter(img, (lo + hi) / 2)


if __name__ == "__main__":
    img = make_scene()

    TARGET = 28.0  # dB; both degradations forced to ~this PSNR
    noisy = match_psnr_noise(img, TARGET)
    blurred = match_psnr_blur(img, TARGET)

    dr = 255.0
    rows = [
        ("noise", noisy),
        ("blur", blurred),
    ]
    scores = {}
    print(f"{'distortion':>10} | {'PSNR (dB)':>10} | {'SSIM':>6}")
    print("-" * 34)
    for name, deg in rows:
        s = ssim(img, deg, data_range=dr)
        scores[name] = s
        print(f"{name:>10} | {psnr(img, deg):>10.2f} | {s:>6.3f}")

    worse = min(scores, key=scores.get)
    print(f"\nPSNR rates the two nearly equal, but SSIM does not: the '{worse}' "
          f"version\nscores notably lower ({scores[worse]:.3f}). A single "
          "pixel-error number cannot\ntell which looks worse -- a structure-aware "
          "metric can. That gap is\nexactly why Netflix fuses several perceptual "
          "features into VMAF.")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, (title, im) in zip(ax, [("original", img)] + rows):
        a.imshow(im, cmap="gray", vmin=0, vmax=255)
        if title == "original":
            a.set_title("original")
        else:
            a.set_title(f"{title}\nPSNR {psnr(img, im):.1f} dB | "
                        f"SSIM {ssim(img, im, data_range=dr):.3f}")
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Equal PSNR, unequal perception -- the gap VMAF was built to close",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("day34_psnr_vs_perception.png", dpi=130, bbox_inches="tight")
    print("\nsaved day34_psnr_vs_perception.png")