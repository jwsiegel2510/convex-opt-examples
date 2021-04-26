# Author: Jonathan Siegel
#
# Contains an example of image denoising using the total variation image denosing model.
# Specifically, this model is given by solving
# 
# min_x |x-f|^2 + lambda * |nabla x|_1
#
# where f is the noisy image and lambda is a parameter controlling the tradeoff between
# the l1-norm of the gradient and the distance to the noisy image f.
# 
# This problem is solved by applying accelerated FB-splitting to the dual problem.

from utils import load_image, plot_image
import numpy as np
from numpy.random import random_sample as rand
from algorithms import total_variation_denoising

noise_level = 1.1

def main():
    # Load and plot original image.
    print('Load and displaying original image.')
    image = load_image()
    plot_image(image)

    # Add Gaussian noise to the image and clip it back to [0,1].
    print('\nImage has been corrupted and made noisy.')
    image = np.clip(image + np.random.normal(scale=noise_level, size=image.shape), 0, 1)
    plot_image(image)

    # Reconstruct the image by optimizing the total variation funcional.
    print('\nPerforming image denoising using total variation.')
    rec_image = total_variation_denoising(image, lam=0.0011, num_iter=200)
    print('Denoising complete!')
    print('Displaying Denoised Image.')
    plot_image(np.clip(rec_image, 0, 1))

if __name__ == '__main__':
    main()
