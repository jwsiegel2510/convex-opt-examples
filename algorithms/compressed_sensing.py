# Author: Jonathan Siegel
#
# Implements code which calculates the Danzig estimator, given by
#
# argmin_{Ax=b} |x|_1,
#
# where A selects a portion of the Fourier transform of x. This problem
# can be used to recover sparse vectors in compressed sensing.
#
# The algorithm which is implemented to solve this is the Douglas-Rachford
# iteration.

import numpy as np

# Calculate the orthogonal projection onto the constraint set.
def fourier_project(x, transform, mask):
    x_transform = np.fft.rfft(x)
    new_transform = np.multiply(mask, transform) + np.multiply(1.0 - mask, x_transform)
    return np.fft.irfft(new_transform)

# Calculate the l1-proximal map, i.e. soft thresholding.
def soft_threshold(x, lam):
    return np.maximum(0, x - lam) + np.minimum(0, x + lam)

# Solve the problem using Douglas-Rachford splitting.
def sparse_recover(sampled_transform, mask, step = 0.5, iter_count=100, verbose=True):
    x = np.fft.irfft(sampled_transform)
    x_temp = x
    for i in range(iter_count):
        x_prev = x_temp
        # Perform a reflected prox for the l1-norm.
        x_temp = 2.0 * soft_threshold(x_temp, step) - x_temp
        # Perform a reflected prox for the constraint and store midpoint.
        x = fourier_project(x_temp, sampled_transform, mask)
        x_temp = 2.0 * x - x_temp
        # Average with the identity.
        x_temp = 0.5 * x_temp + 0.5 * x_prev
        # Calculate and report the loss.
        if verbose:
            print('The loss is at iteration %d is %f' % (i, np.sum(np.abs(x))))
    return x
