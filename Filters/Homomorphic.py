import cv2
import numpy as np
import matplotlib.pyplot as plt

def homomorphic_filter(image_path, D0=50, GL=0.9, GH=1.9):
	image = cv2.imread(image_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Display the input image in grayscale
	plt.subplot(2, 3, 1)
	plt.imshow(gray, cmap='gray')
	plt.title('Input Image') 

	# Convert image to double
	gray = gray.astype(float)
	b = gray.copy()

	[m, n] = gray.shape   # size of image 
	# Add 1 to pixels to avoid issues with 0 values
	b = b + 1

	# Take the natural logarithm to each pixel value of the image
	log_b = np.log(b)

	# Take the Fourier transform
	fourier = np.fft.fft2(log_b)

	# Shift the zero frequency component to the center
	dd = np.fft.fftshift(fourier)

	# Initialize the Homomorphic filter
	H = np.zeros_like(gray)
	for u in range(m):
		for v in range(n):
			H[u, v] = (GH - GL) * (1 - np.exp(-1 * (np.sqrt((u - m / 2) ** 2 + (v - n / 2) ** 2)) ** 2 / D0) ** 2) + GL


	# applies the homomorphic filter to the frequency domain
	x = dd * H  #multiplication Shifted Fourier transform with homomorphic filter


	# Perform the inverse Fourier transform
	real_x = np.abs(np.fft.ifft2(x))

	# Display the inverse Fourier transform
	plt.subplot(2, 3, 2)
	plt.imshow(real_x, cmap='gray')
	plt.title('Inverse Fourier Transform')

	# Exponential transformation
	final_result = np.exp(real_x)

	return final_result.astype(np.uint8)






