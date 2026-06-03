import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("AI-headshot.png", 0)

# FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = rows//2 , cols//2

# Create mask
mask = np.zeros((rows, cols), np.uint8)
r = 50
mask[crow-r:crow+r, ccol-r:ccol+r] = 1

# Apply mask
fshift_filtered = fshift * mask

# Inverse FFT
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Plot
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(img_back, cmap='gray')
plt.title("Low Pass Filtered")

plt.show()