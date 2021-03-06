# Author: Jonathan Siegel
#
# Contains a rudimentary inpainting and denoising example to play around with.

from utils import load_image, plot_image
import numpy as np
from numpy.random import random_sample as rand
from algorithms import reconstruct, trivial_reconstruct

loss_frac = 0.9

def main():
    # Load and plot original image.
    print('Load and displaying original image.')
    image = load_image()
    plot_image(image)

    # Randomly delete pixels from the original image.
    print('\nMany pixel values have been lost.\nHere is the corrupted image.')
    mask = np.double(rand(np.shape(image)) > loss_frac)
    corr_image = np.multiply(image, mask)
    plot_image(corr_image)

    # Reconstruct the image
    print('\nPerforming image reconstruction.')
    rec_image = trivial_reconstruct(corr_image, mask) #, lam = 0.03, max_iter=100)
    print('Reconstruction complete!')
    print('Displaying Reconstructed Image.')
    plot_image(rec_image)

if __name__ == '__main__':
    main()
