import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image in grayscale
image = cv2.imread("/home/sidharth/Documents/rotation_data/train/0/0aDBJDO9rE.jpg", cv2.IMREAD_GRAYSCALE)

# Compute the 2D Fourier Transform of the image
f_transform = np.fft.fft2(image)

# Shift the zero-frequency component to the center
f_shift = np.fft.fftshift(f_transform)

# Compute the magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)  # Add 1 to avoid log(0)

# Display the original image and its Fourier spectrum
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.show()

# # Reconstruct the image using the Inverse Fourier Transform
# f_ishift = np.fft.ifftshift(f_shift)
# reconstructed_image = np.fft.ifft2(f_ishift)
# reconstructed_image = np.abs(reconstructed_image)

# # Display the reconstructed image
# plt.figure(figsize=(5, 5))
# plt.title("Reconstructed Image")
# plt.imshow(reconstructed_image, cmap='gray')
# plt.axis('off')
# plt.show()
