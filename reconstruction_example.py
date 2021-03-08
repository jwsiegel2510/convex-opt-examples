# Author: Jonathan Siegel
#
# Contains an image reconstruction example, where an image is reconstructed from
# a fraction of its Fourier transform. The reconstruction is performed by finding
# a minimal wavelet l1 norm image which matches the gives Fourier coefficients 
# (danzig estimator) or by solving a corresponding lasso problem..

from utils import load_image, plot_image
import numpy as np
from numpy.random import random_sample as rand
from algorithms import reconstruct_danzig, reconstruct_lasso

sample_frac = 0.50

def main():
    # Load and plot original image.
    print('Load and displaying original image.')
    image = load_image(filename='lenna-image.jpeg')
    plot_image(image)

    # Randomly sample entries of the Fourier transform of the signal.
    print('\n%2.0f%% of the Fourier transform of the image has been sampled.' % (sample_frac*100))
    transform = np.fft.rfftn(image, norm='ortho')
    mask = np.double(rand(transform.shape) < sample_frac)
    sampled_transform = np.multiply(transform, mask)
    print('Displaying image obtained by setting remaining Fourier modes to 0.')
    print('Note that this image fits the samples while minimizing the wavelet l2 norm.')
    plot_image(np.fft.irfftn(sampled_transform, norm='ortho'))

    # Reconstruct the image
    print('\nPerforming image reconstruction.')
    rec_image = reconstruct_danzig(sampled_transform, mask, step=0.4, max_iter=50, wavelet_order=2)
    print('Reconstruction complete!')
    print('Displaying Reconstructed Image.')
    plot_image(rec_image)

if __name__ == '__main__':
    main()
