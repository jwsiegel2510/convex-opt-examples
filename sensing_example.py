# Author: Jonathan Siegel
#
# Here we implement a simple test of compressed sensing. We generate a 
# sparse vector and sample a portion of its Fourier transform. We then
# reconstruct the vector using the Danzig estimator.

import math
import numpy as np
from algorithms import sparse_recover

sparsity_fraction = 0.01
sampling_fraction = 0.1
vector_length = 512

def main():
    # Generate a random sparse vector.
    print('Generating a random vector with %d%% nonzero entries' % (sparsity_fraction * 100))
    x = np.random.normal(size=vector_length)
    nonzero_count = math.floor(sparsity_fraction * vector_length)
    indices = np.random.permutation(vector_length)
    for i in range(nonzero_count, vector_length):
        x[indices[i]] = 0
    print(len(np.nonzero(x)[0]))

    # Generate Fourier mask and sample Fourier transform
    print('Randomly sampling %d%% of its Fourier transform' % (sampling_fraction * 100))
    transform = np.fft.rfft(x)
    mask = np.zeros(transform.size)
    nonzero_mask = math.floor(sampling_fraction * transform.size)
    indices = np.random.permutation(transform.size)
    for i in range(nonzero_mask):
        mask[indices[i]] = 1
    print(len(np.nonzero(mask)[0]))
    transform = np.multiply(transform, mask)

    # Reconstruct the signal from the sampled transform and compare with the original.
    x_rec = sparse_recover(transform, mask, step=1.0, iter_count=500, verbose=False)
    print('The relative reconstruction error is %f' % (np.linalg.norm(x - x_rec) / np.linalg.norm(x)))

if __name__ == '__main__':
    main()
