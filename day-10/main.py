"""
Day 10 — Eigenvectors & Eigenfaces
All Things Visual Roadmap · Phase 1: Mathematical Foundations

Exercise: Build an eigenface decomposition from scratch using the
Olivetti Faces dataset (bundled with scikit-learn).

Requirements:
    pip install numpy matplotlib scikit-learn

Run:
    python day10_eigenvectors_eigenfaces.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

import certifi, ssl, os
os.environ["SSL_CERT_FILE"] = certifi.where()

# ─────────────────────────────────────────────
# 1. Load the dataset
# ─────────────────────────────────────────────
print("Loading Olivetti Faces dataset...")
data = fetch_olivetti_faces(shuffle=False)
faces = data.data          # shape: (400, 4096)  — each row is a 64×64 face flattened
n_samples, n_pixels = faces.shape
image_shape = (64, 64)
print(f"Loaded {n_samples} faces, each {image_shape[0]}x{image_shape[1]} = {n_pixels} pixels")

# ─────────────────────────────────────────────
# 2. Compute and visualize the mean face
# ─────────────────────────────────────────────
mean_face = faces.mean(axis=0)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.imshow(mean_face.reshape(image_shape), cmap="gray")
ax.set_title("Mean Face")
ax.axis("off")
plt.tight_layout()
plt.savefig("day10_mean_face.png", dpi=100)
plt.show()

# ─────────────────────────────────────────────
# 3. Center the data (subtract the mean)
# ─────────────────────────────────────────────
centered = faces - mean_face  # shape: (400, 4096)

# ─────────────────────────────────────────────
# 4. Compute covariance matrix & eigenvectors
# ─────────────────────────────────────────────
# The "trick" from Turk & Pentland (1991):
# Instead of the huge 4096×4096 covariance matrix (C = X^T X),
# compute the smaller 400×400 matrix (L = X X^T), find its
# eigenvectors, then project back to get the real eigenvectors.

print("Computing eigenvectors (this may take a few seconds)...")
L = centered @ centered.T  # shape: (400, 400)
eigenvalues_small, eigenvectors_small = np.linalg.eigh(L)

# eigh returns in ascending order — flip to descending
idx = np.argsort(eigenvalues_small)[::-1]
eigenvalues_small = eigenvalues_small[idx]
eigenvectors_small = eigenvectors_small[:, idx]

# Project back to the original pixel space to get eigenfaces
eigenfaces = eigenvectors_small.T @ centered  # shape: (400, 4096)

# Normalize each eigenface to unit length
norms = np.linalg.norm(eigenfaces, axis=1, keepdims=True)
norms[norms == 0] = 1  # avoid division by zero
eigenfaces = eigenfaces / norms

# Eigenvalues (proportional — the exact scale doesn't matter for ordering)
eigenvalues = eigenvalues_small
total_variance = eigenvalues.sum()
print(f"Total variance: {total_variance:.2f}")

# ─────────────────────────────────────────────
# 5. Visualize the top 16 eigenfaces
# ─────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenfaces[i].reshape(image_shape), cmap="gray")
    var_pct = 100 * eigenvalues[i] / total_variance
    ax.set_title(f"EF {i+1} ({var_pct:.1f}%)", fontsize=9)
    ax.axis("off")
fig.suptitle("Top 16 Eigenfaces", fontsize=14)
plt.tight_layout()
plt.savefig("day10_top16_eigenfaces.png", dpi=100)
plt.show()

# ─────────────────────────────────────────────
# 6. Reconstruct a face with varying # of eigenfaces
# ─────────────────────────────────────────────
test_idx = 42  # pick any face
test_face = centered[test_idx]  # already mean-subtracted

n_components_list = [10, 25, 50, 100, 200]

fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(16, 3))

# Original
axes[0].imshow(faces[test_idx].reshape(image_shape), cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

for j, n_comp in enumerate(n_components_list):
    # Project onto the first n_comp eigenfaces
    weights = test_face @ eigenfaces[:n_comp].T   # shape: (n_comp,)
    reconstruction = mean_face + weights @ eigenfaces[:n_comp]

    axes[j + 1].imshow(reconstruction.reshape(image_shape), cmap="gray")
    axes[j + 1].set_title(f"{n_comp} components")
    axes[j + 1].axis("off")

fig.suptitle(f"Reconstruction of Face #{test_idx}", fontsize=14)
plt.tight_layout()
plt.savefig("day10_reconstruction.png", dpi=100)
plt.show()

# ─────────────────────────────────────────────
# 7. Plot reconstruction error vs # of eigenfaces
# ─────────────────────────────────────────────
max_comp = min(200, len(eigenfaces))
errors = []
for k in range(1, max_comp + 1):
    weights = test_face @ eigenfaces[:k].T
    recon = mean_face + weights @ eigenfaces[:k]
    err = np.mean((faces[test_idx] - recon) ** 2)
    errors.append(err)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Reconstruction error
ax1.plot(range(1, max_comp + 1), errors, linewidth=2)
ax1.set_xlabel("Number of Eigenfaces")
ax1.set_ylabel("Mean Squared Error")
ax1.set_title("Reconstruction Error vs. Components")
ax1.grid(True, alpha=0.3)

# Cumulative variance explained
cumulative_var = np.cumsum(eigenvalues[:max_comp]) / total_variance * 100
ax2.plot(range(1, max_comp + 1), cumulative_var, linewidth=2, color="green")
ax2.axhline(y=95, color="red", linestyle="--", alpha=0.7, label="95% variance")
ax2.set_xlabel("Number of Eigenfaces")
ax2.set_ylabel("Cumulative Variance Explained (%)")
ax2.set_title("Variance Captured vs. Components")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("day10_error_and_variance.png", dpi=100)
plt.show()

# Print key stats
n_95 = np.searchsorted(cumulative_var, 95) + 1
print(f"\nKey findings:")
print(f"  Top 50 eigenfaces capture {cumulative_var[49]:.1f}% of total variance")
print(f"  You need {n_95} eigenfaces to capture 95% of variance")
print(f"  That's {n_95}/{n_pixels} = {100*n_95/n_pixels:.1f}% of the original dimensions")
print(f"\nThat's massive compression — and the face is still recognizable!")