import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

img = cv2.imread("AI-headshot.png", 0)

# Resize for simplicity
img = cv2.resize(img, (256,256))

block_size = 8

compressed = np.zeros_like(img, dtype=np.float32)

for i in range(0,256,block_size):
    for j in range(0,256,block_size):

        block = np.float32(img[i:i+8, j:j+8])

        # DCT
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        # Quantization simulation
        dct_block[4:,4:] = 0

        # Inverse DCT
        reconstructed = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

        compressed[i:i+8, j:j+8] = reconstructed

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(compressed, cmap='gray')
plt.title("DCT Compressed")

plt.show()