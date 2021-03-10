# Author: Jonathan Siegel
#
# Contains a simple wavelet compression example, designed to demonstrate the remarkable effectiveness
# of wavelets in compressing natural images.

from utils import load_image, plot_image
import numpy as np
from algorithms import wavelet_transform_2d, inv_wavelet_transform_2d
import math

nonzero_fraction = 0.01

def main():
    # Load and plot original image.
    print('Load and displaying original image.')
    image = load_image()
    plot_image(image)

    # Calculate the wavelet transform and keep the largest coefficients.
    print('\nWe keep only %2.0f%% of the wavelet coefficients' % (nonzero_fraction * 100))
    transform = wavelet_transform_2d(image, wavelet_order=2)
    k = math.floor(transform.size * nonzero_fraction)
    cutoff = np.partition(np.abs(transform).flatten(), -k)[-k]
    transform = np.multiply(transform, np.double(np.abs(transform) >= cutoff))
    print(transform.size, k, np.count_nonzero(transform))

    # Reconstruct the image
    print('\nReconstructing image from remaining wavelets.')
    rec_image = inv_wavelet_transform_2d(transform, wavelet_order=2)
    print('Displaying Compressed Image.')
    # Assumes that the size of a float is 8 bits.
    compression_factor = nonzero_fraction - (1/8)*(math.log2(nonzero_fraction) * nonzero_fraction + \
                                                   math.log2(1.0 - nonzero_fraction) * (1.0 - nonzero_fraction))
    print('Image has been compressed by a factor of %f' % (1.0 / compression_factor))
    plot_image(np.clip(rec_image, 0, 1))

if __name__ == '__main__':
    main()
