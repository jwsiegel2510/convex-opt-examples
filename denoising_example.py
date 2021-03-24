# Author: Jonathan Siegel
#
# Contains an example of image denoising using the proximal map of the l1-norm of the wavelet coefficients.

from utils import load_image, plot_image
import numpy as np
from numpy.random import random_sample as rand
from algorithms import wavelet_l1_prox

noise_level = 1.0

def main():
    # Load and plot original image.
    print('Load and displaying original image.')
    image = load_image()
    plot_image(image)

    # Add Gaussian noise to the image and clip it back to [0,1].
    print('\nImage has been corrupted and made noisy.')
    image = image + np.random.normal(scale=noise_level, size=image.shape)
    plot_image(np.clip(image, 0, 1))

    # Reconstruct the image by soft-thresholding the wavelet coefficients.
    print('\nPerforming image denoising using wavelets.')
    rec_image = wavelet_l1_prox(image, lam=1.3, wavelet_order=3)
    print('Denoising complete!')
    print('Displaying Denoised Image.')
    plot_image(np.clip(rec_image, 0, 1))

if __name__ == '__main__':
    main()
