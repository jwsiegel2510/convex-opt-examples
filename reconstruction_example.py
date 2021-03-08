# Author: Jonathan Siegel
#
# Contains a rudimentary inpainting example to play around with.

from utils import load_image, plot_image
import numpy as np
from numpy.random import random_sample as rand
from algorithms import reconstruct, trivial_reconstruct

def main():
    # Load and plot original image.
    print('Load and displaying original image.')
    image = load_image()
    plot_image(image)

    # Randomly delete pixels from the original image.
    print('\nMany pixel values have been lost.\nHere is the corrupted image.')
    mask = np.ones_like(image)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if i%40 < 30 and j % 40 < 30:
                mask[i][j] = 0
    corr_image = np.multiply(image, mask)
    plot_image(corr_image)

    # Reconstruct the image
    print('\nPerforming image reconstruction.')
    rec_image = reconstruct(corr_image, mask, lam = 0.3, max_iter=50, alg='dr', wavelet_order=4)
    print('Reconstruction complete!')
    print('Displaying Reconstructed Image.')
    plot_image(rec_image)

if __name__ == '__main__':
    main()
