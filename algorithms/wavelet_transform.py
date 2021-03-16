# Author: Jonathan Siegel
#
# Implements the wavelet transform, the wavelet l1 prox, and the wavelet l1 norm.
# For simplicity, all wavelet transforms are done with periodic boundary conditions.

import numpy as np
from numpy.random import rand
from numpy.polynomial import Polynomial as P
import math

# Calculate and cache wavelet filters. Uses polynomial roots, reasonably stable up to order 10.
principal_filters = {}
wavelet_filters = {}
def get_filters(order):
    if order not in principal_filters or order not in wavelet_filters:
        # Calculate principal filter.
        principal_filter = None
        if order == 0:
            principal_filter = np.array([1.0, 1.0])
            principal_filter /= np.linalg.norm(principal_filter)
        else:
            # Calculate binomial coefficients.
            bin_coeff = np.zeros([order + 2])
            bin_coeff[0] = 1.0
            for i in range(order + 1):
                for j in range(order, -1, -1):
                    bin_coeff[j+1] += bin_coeff[j]
            # Set up coefficient matrix.
            matrix = np.matrix(np.zeros([order + 1, order + 1]))
            for i in range(order + 1):
                for j in range(1 + 2 * i):
                    if j < order + 2 and 2*i - j < order + 1:
                        matrix[i, 2*i - j] += bin_coeff[j]
            # Solve for the coefficients of q(z)*q(z^-1)
            target = np.zeros([order + 1])
            target[0] = 1.0
            coeffs = np.linalg.solve(matrix, target)
            # Construct polynomial and find roots.
            poly = P(coeffs)
            roots = poly.roots()
            # Determine filter based on roots.
            principal_filter = np.zeros([2 * order + 2], dtype=complex)
            principal_filter[0] = 1
            for root in roots:
                # Only consider real roots and imaginary roots with positive real part.
                if root.imag > -1e-10:
                    z = root + np.sqrt(root ** 2 - 1)
                    for j in range(2 * order + 1, 0, -1):
                        principal_filter[j] = principal_filter[j-1] - z * principal_filter[j]
                    principal_filter[0] *= (-z)
                # If the root is imaginary, take the conjugate of z as well.
                if root.imag > 1e-10:
                    z = np.conjugate(z)
                    for j in range(2 * order + 1, 0, -1):
                        principal_filter[j] = principal_filter[j-1] - z * principal_filter[j]
                    principal_filter[0] *= (-z)
            principal_filter = principal_filter.real
            for i in range(order + 1):
                for j in range(2 * order + 1, 0, -1):
                    principal_filter[j] = principal_filter[j-1] + principal_filter[j]
            principal_filter /= np.linalg.norm(principal_filter)
        # Obtain the wavelet filter by reversing and alternating signs.
        wavelet_filter = np.zeros_like(principal_filter)
        for j in range(principal_filter.size):
            wavelet_filter[j] = principal_filter[principal_filter.size - j - 1]
            if j%2 == 1:
                wavelet_filter[j] *= -1
        principal_filters[order] = principal_filter
        wavelet_filters[order] = wavelet_filter
    return {'principal': principal_filters[order], 'wavelet': wavelet_filters[order]}

# Wavelet transform. Assumes the signal length is a power of 2. Performs the transform for each row.
def wavelet_transform_1d(signal, wavelet_order=0):
    if signal.shape[1] <= 1:
        return signal
    if signal.shape[1] % 2 != 0:
        print('An error occurred in wavelet transform. Size not a power of 2.')
    transform = np.zeros_like(signal)
    filters = get_filters(wavelet_order)
    principal_filter = filters['principal']
    wavelet_filter = filters['wavelet']
    for j in range(principal_filter.size):
        transform[:, 0:(signal.shape[1] // 2)] += principal_filter[j] * np.roll(signal, -j, axis=1)[:, 0::2]
        transform[:, (signal.shape[1] // 2):] += wavelet_filter[j] * np.roll(signal, -j, axis=1)[:, 0::2]
    transform[:, 0:signal.shape[1] // 2] = wavelet_transform_1d(transform[:, 0:signal.shape[1] // 2], wavelet_order)
    return transform

# Inverse wavelet transform. Assumes the signal length is a power of 2. Performs the inverse transform for each row.
def inv_wavelet_transform_1d(transform, wavelet_order=0):
    if transform.shape[1] <= 1:
        return transform
    if transform.shape[1] % 2 != 0:
        print('An error occurred in wavelet transform. Size not a power of 2.')
    transform[:, 0:(transform.shape[1] // 2)] = inv_wavelet_transform_1d(transform[:, 0:(transform.shape[1] // 2)], wavelet_order)
    signal = np.zeros_like(transform)
    filters = get_filters(wavelet_order)
    principal_filter = filters['principal']
    wavelet_filter = filters['wavelet']
    for i in range(signal.shape[1]):
        for j in range(principal_filter.size // 2):
            index = (((i // 2) - j)%(transform.shape[1] // 2) + (transform.shape[1] // 2)) % (transform.shape[1] // 2)
            signal[:, i] += principal_filter[(i%2) + 2*j] * transform[:, index]
            signal[:, i] += wavelet_filter[(i%2) + 2*j] * transform[:, index + (transform.shape[1] // 2)]
    return signal

# Wavelet transform in 2D. Assumes both image dimensions are a power of 2.
def wavelet_transform_2d(image, wavelet_order=0):
    image = wavelet_transform_1d(image, wavelet_order)
    image = np.transpose(wavelet_transform_1d(np.transpose(image), wavelet_order))
    return image

# Inverse wavelet transform in 2D. Assumes both image dimension are a power of 2.
def inv_wavelet_transform_2d(transform, wavelet_order=0):
    transform = np.transpose(inv_wavelet_transform_1d(np.transpose(transform), wavelet_order))
    transform = inv_wavelet_transform_1d(transform, wavelet_order)
    return transform

# Calculate the l1 norm of the wavelet coefficients.
def wavelet_l1(image, wavelet_order=0):
    transform = wavelet_transform_2d(image, wavelet_order)
    return np.sum(np.abs(transform))

# HERE IS THE FUNCTION YOU NEED TO IMPLEMENT. Calculate the wavelet-l1 prox.
def wavelet_l1_prox(image, wavelet_order=0, lam=1.0):

# Tests the wavelet transforms.
def main():
    # k is the order to test.
    k = 10
    filters = get_filters(k)
    print(filters)
    wavelet_filter = filters['wavelet']
    vect = np.ones_like(wavelet_filter)
    for i in range(k+1):
        print(np.inner(vect, wavelet_filter))
        for j in range(2*k+2):
            vect[j] *= (j-i)
    array = rand(512, 512)
    transform = wavelet_transform_2d(array, wavelet_order=k)
    print(np.linalg.norm(array))
    print(np.linalg.norm(transform))
    print(np.linalg.norm(inv_wavelet_transform_2d(transform, wavelet_order=k) - array))

if __name__ == '__main__':
    main()
