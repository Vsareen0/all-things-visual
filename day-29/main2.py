from PIL import Image
import numpy as np
from collections import Counter
import math
import matplotlib.pyplot as plt

# Load image
img = Image.open("image.jpg").convert("L")  # grayscale
pixels = np.array(img, dtype=np.int16)

# Original size estimate
original_bits = pixels.size * 8

# Predictive coding (left neighbor prediction)
residual = np.zeros_like(pixels)

residual[:, 0] = pixels[:, 0]

for y in range(pixels.shape[0]):
    for x in range(1, pixels.shape[1]):
        residual[y, x] = pixels[y, x] - pixels[y, x - 1]

# Entropy calculation
def entropy(arr):
    counts = Counter(arr.flatten())
    total = sum(counts.values())

    H = 0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)

    return H

orig_entropy = entropy(pixels)
res_entropy = entropy(residual)

orig_estimated_bits = orig_entropy * pixels.size
res_estimated_bits = res_entropy * pixels.size

reduction = 100 * (1 - res_estimated_bits / orig_estimated_bits)

print(f"Original entropy: {orig_entropy:.2f} bits/pixel")
print(f"Residual entropy: {res_entropy:.2f} bits/pixel")
print(f"Estimated reduction: {reduction:.2f}%")



plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(pixels, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(np.abs(residual), cmap="gray")
plt.title("Residual Image (Prediction Errors)")
plt.axis("off")

plt.show()


print(f"Original entropy: {orig_entropy:.2f} bits/pixel")
print(f"Residual entropy: {res_entropy:.2f} bits/pixel")

improvement = (1 - res_entropy/orig_entropy)*100

print(f"Entropy reduction: {improvement:.2f}%")