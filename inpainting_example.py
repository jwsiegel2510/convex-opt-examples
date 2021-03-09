# Author: Jonathan Siegel
#
# Contains an image reconstruction example, where an image is reconstructed from
# a fraction of its Fourier transform. The reconstruction is performed by finding
# a minimal wavelet l1 norm image which matches the gives Fourier coefficients 
# (danzig estimator) or by solving a corresponding lasso problem..

from utils import load_image, plot_image
import numpy as np
from numpy.random import random_sample as rand
from algorithms import inpaint_danzig, inpaint_lasso, trivial_inpaint

sample_frac = 0.3

def main():
    # Load and plot original image.
    print('Load and displaying original image.')
    image = load_image()
    plot_image(image)

    # Randomly sample entries of the Fourier transform of the signal.
    print('\n%2.0f%% of the image has been sampled.' % (sample_frac*100))
    mask = np.double(rand(image.shape) < sample_frac)
    sampled_image = np.multiply(mask, image)
    plot_image(sampled_image)

    # Reconstruct the image
    print('\nPerforming image reconstruction.')
    rec_image = inpaint_danzig(sampled_image, mask, step=0.3, max_iter=50, wavelet_order=2)
    print('Reconstruction complete!')
    print('Displaying Reconstructed Image.')
    plot_image(np.clip(rec_image, 0, 1))

if __name__ == '__main__':
    main()
