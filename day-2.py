"""
Day 2 — Matrices as Geometric Transformations
All Things Visual Roadmap · Phase 1 · Linear Algebra for Vision

Exercise: Build transformation matrices by hand, apply them to an image,
and discover why composition order matters.

Prerequisites: pip install numpy matplotlib Pillow
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================================================
# PART 1: Visualize what a 2x2 matrix does to the unit square
# ============================================================

def plot_transform(ax, matrix, title):
    """Apply a 2x2 matrix to the unit square and plot before/after."""
    # Unit square corners: origin, right, top-right, top
    square = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],  # close the shape
    ]).T  # shape (2, 5)

    transformed = matrix @ square

    ax.fill(square[0], square[1], alpha=0.2, color="blue", label="Original")
    ax.plot(square[0], square[1], "b-", linewidth=1.5)

    ax.fill(transformed[0], transformed[1], alpha=0.3, color="red", label="Transformed")
    ax.plot(transformed[0], transformed[1], "r-", linewidth=2)

    # Draw basis vectors
    origin = np.array([0, 0])
    col1 = matrix[:, 0]
    col2 = matrix[:, 1]
    ax.annotate("", xy=col1, xytext=origin,
                arrowprops=dict(arrowstyle="->", color="darkred", lw=2))
    ax.annotate("", xy=col2, xytext=origin,
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2))

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)


# Define several transformation matrices
theta = np.radians(45)  # 45-degree rotation

transforms = {
    "Identity": np.array([[1, 0], [0, 1]], dtype=float),
    "Scale 2x horizontal": np.array([[2, 0], [0, 1]], dtype=float),
    "Uniform scale 0.5x": np.array([[0.5, 0], [0, 0.5]], dtype=float),
    f"Rotate {int(np.degrees(theta))} deg": np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ]),
    "Shear (horizontal)": np.array([[1, 0.5], [0, 1]], dtype=float),
    "Reflection (y-axis)": np.array([[-1, 0], [0, 1]], dtype=float),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("What 2x2 Matrices Do to the Unit Square", fontsize=14, fontweight="bold")

for ax, (name, mat) in zip(axes.ravel(), transforms.items()):
    plot_transform(ax, mat, name)

plt.tight_layout()
plt.savefig("day02_unit_square_transforms.png", dpi=150)
plt.show()
print("Saved: day02_unit_square_transforms.png")

# ============================================================
# PART 2: Apply transformations to an actual image
# ============================================================

def make_test_image(size=200):
    """Create a simple test image with an asymmetric pattern so you can
    visually tell when it's been rotated/flipped/sheared."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = [30, 30, 30]  # dark gray background

    # Red triangle in top-left quadrant
    for y in range(size // 2):
        for x in range(y):
            img[y, x] = [220, 50, 50]

    # Blue rectangle in bottom-right
    img[size*3//4:size-10, size//2:size-10] = [50, 80, 220]

    # Green dot near center
    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[-cy:size-cy, -cx:size-cx]
    mask = xx**2 + yy**2 < 15**2
    img[mask] = [50, 200, 80]

    return img


def apply_affine_to_image(img, matrix_2x2, output_size=None):
    """Apply a 2x2 linear transform to an image using inverse mapping.

    This is how image warping actually works under the hood:
    for each output pixel, compute where it came FROM in the input,
    then sample that location. This avoids holes in the output.
    """
    h, w = img.shape[:2]
    out_h, out_w = output_size or (h, w)
    output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Center of the image (we transform around the center, not the corner)
    cx_in, cy_in = w / 2, h / 2
    cx_out, cy_out = out_w / 2, out_h / 2

    # Inverse of the transformation matrix
    inv_mat = np.linalg.inv(matrix_2x2)

    for out_y in range(out_h):
        for out_x in range(out_w):
            # Shift to center, apply inverse transform, shift back
            p = np.array([out_x - cx_out, out_y - cy_out])
            src = inv_mat @ p
            src_x = int(round(src[0] + cx_in))
            src_y = int(round(src[1] + cy_in))

            if 0 <= src_x < w and 0 <= src_y < h:
                output[out_y, out_x] = img[src_y, src_x]

    return output


print("\nGenerating test image and applying transforms...")
test_img = make_test_image(150)

# Pick a few transforms to apply to the image
image_transforms = {
    "Original": np.eye(2),
    "Rotate 30 deg": np.array([
        [np.cos(np.radians(30)), -np.sin(np.radians(30))],
        [np.sin(np.radians(30)),  np.cos(np.radians(30))]
    ]),
    "Scale 1.5x": np.array([[1.5, 0], [0, 1.5]]),
    "Shear": np.array([[1, 0.4], [0, 1]]),
}

fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
fig2.suptitle("Transformations Applied to an Image", fontsize=13, fontweight="bold")

for ax, (name, mat) in zip(axes2, image_transforms.items()):
    result = apply_affine_to_image(test_img, mat, output_size=(200, 200))
    ax.imshow(result)
    ax.set_title(name, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("day02_image_transforms.png", dpi=150)
plt.show()
print("Saved: day02_image_transforms.png")

# ============================================================
# PART 3: Composition order matters!
# ============================================================

print("\n--- PART 3: Does order matter? ---")

# Define two transforms
S = np.array([[2, 0], [0, 1]], dtype=float)   # scale x by 2
R = np.array([
    [np.cos(np.radians(45)), -np.sin(np.radians(45))],
    [np.sin(np.radians(45)),  np.cos(np.radians(45))]
])  # rotate 45 degrees

# Compose in two different orders
scale_then_rotate = R @ S  # S applied first, then R
rotate_then_scale = S @ R  # R applied first, then S

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5))
fig3.suptitle("Composition Order Matters!", fontsize=13, fontweight="bold")

plot_transform(axes3[0], S, "Just Scale (2x horizontal)")
plot_transform(axes3[1], scale_then_rotate, "Scale THEN Rotate (R @ S)")
plot_transform(axes3[2], rotate_then_scale, "Rotate THEN Scale (S @ R)")

plt.tight_layout()
plt.savefig("day02_composition_order.png", dpi=150)
plt.show()
print("Saved: day02_composition_order.png")

# ============================================================
# QUESTIONS TO ANSWER
# ============================================================
print("""
--- Questions ---

1. Look at Part 1. For the rotation matrix, verify that both columns
   have length 1 (they're unit vectors). What does this mean geometrically?
   Hint: rotations preserve distances.

    - The rotation matrix for 45° is [[cos45, -sin45], [sin45, cos45]]. Both columns 
    have length sqrt(cos²θ + sin²θ) = sqrt(1) = 1 — this is just the Pythagorean identity, 
    so it holds for any angle. Geometrically, this means rotation preserves all distances. 
    The unit square doesn't stretch or shrink — it only spins. Any matrix whose columns are 
    orthogonal unit vectors is called an orthogonal matrix, and these are exactly the distance-preserving 
    linear transforms.

2. Look at Part 2. The image warp function uses INVERSE mapping.
   Why can't we just iterate over input pixels and map them forward?
   What artifact would that create? (Think about gaps/holes.)

   - With forward mapping, you iterate over input pixels and compute 
   where each one lands in the output. The problem: transformed coordinates 
   rarely land on exact integer grid positions. Some output pixels get multiple 
   inputs piled on them, while others get nothing — leaving black holes. 
   Inverse mapping flips the logic: for each output pixel, ask "where did you come from?" 
   by applying the inverse transform, then sample the input at that location. 
   Every output pixel gets a value, guaranteed. This is how every real image warping 
   library (OpenCV's warpAffine, PIL's transform) works under the hood.

3. Look at Part 3. Describe in words why scale-then-rotate looks
   different from rotate-then-scale. Which one produces a
   parallelogram? Which produces a rotated rectangle?

   - With scale then rotate (R @ S): you first stretch the unit square into a 2:1 rectangle, 
   then rotate that rectangle by 45°. The result is a rotated rectangle — still a rectangle, just tilted and elongated.
    With rotate then scale (S @ R): you first rotate the square to a diamond, then stretch horizontally. 
    That horizontal stretch distorts the diamond unevenly, producing a parallelogram (a sheared, non-rectangular shape). 
    The difference comes down to which operation sees the "original" shape vs the already-transformed one.

4. (Bonus) What is the determinant of each transform in Part 1?
   What does the determinant tell you about how the AREA of the
   unit square changes? Verify by eyeballing the plots.

   -  

5. (Bonus) Construct a matrix that rotates by 90 degrees AND
   scales by 0.5 in a single matrix. Apply it to the test image.
   
   - The rotation matrix for 90° is [[0, -1], [1, 0]]. Multiply by 0.5 to get: 

    M = np.array([[0, -0.5],
              [0.5,  0]])

    This single matrix simultaneously rotates and shrinks. You can verify: det(M) = 0.25, 
    meaning the output area is 1/4 of the input — exactly what you'd expect from halving both dimensions. 
    To apply it to the test image, just call apply_affine_to_image(test_img, M) from the exercise script.

""")