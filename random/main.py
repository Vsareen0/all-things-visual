import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("AI-headshot.png", 0)

# FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Magnitude spectrum
magnitude = 20 * np.log(np.abs(fshift) + 1)

# Plot
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(magnitude, cmap='gray')
plt.title("FFT Magnitude Spectrum")

plt.show()