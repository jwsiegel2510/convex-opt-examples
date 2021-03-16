# Author: Jonathan Siegel
#
# Contains an example of image denoising using the proximal map of the l1-norm of the wavelet coefficients.

from utils import load_image, plot_image
import numpy as np
from numpy.random import random_sample as rand
from algorithms import wavelet_l1_prox

noise_level = 0.7

def main():
    # Load and plot original image.
    print('Load and displaying original image.')
    image = load_image()
    plot_image(image)

    # Randomly sample entries of the Fourier transform of the signal.
    print('\nImage has been corrupted and made noisy.')
    image = image + np.random.normal(scale=noise_level, size=image.shape)
    plot_image(np.clip(image, 0, 1))

    # Reconstruct the image
    print('\nPerforming image denoising using wavelets.')
    # NOTE: This is the function you need to implement in the file algorithms/wavelet_transform.py
    rec_image = wavelet_l1_prox(image, lam=1.1, wavelet_order=3)
    print('Denoising complete!')
    print('Displaying Denoised Image.')
    plot_image(np.clip(rec_image, 0, 1))

if __name__ == '__main__':
    main()
