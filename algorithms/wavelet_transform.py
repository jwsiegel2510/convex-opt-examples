# Author: Jonathan Siegel
#
# Implements the wavelet transform, the wavelet l1 prox, and the wavelet l1 norm.
# For simplicity, all wavelet transforms are done with periodic boundary conditions.

import numpy as np
from numpy.random import rand
import math

principal_filters = {'haar': np.array([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])}
second_filters = {'haar': np.array([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)])}

# Wavelet transform. Assumes the signal length is a power of 2. Performs the transform for each row.
def wavelet_transform_1d(signal, wavelet_type='haar'):
    if signal.shape[1] <= 1:
        return signal
    if signal.shape[1] % 2 != 0:
        print('An error occurred in wavelet transform. Size not a power of 2.')
    transform = np.zeros_like(signal)
    principal_filter = principal_filters[wavelet_type]
    second_filter = second_filters[wavelet_type]
    for i in range(signal.shape[1] // 2):
        for j in range(principal_filter.size):
            transform[:, i] += principal_filter[j] * signal[:, (2*i + j) % signal.shape[1]]
            transform[:, signal.shape[1] // 2 + i] += second_filter[j] * signal[:, (2*i + j) % signal.shape[1]]
    transform[:, 0:signal.shape[1] // 2] = wavelet_transform_1d(transform[:, 0:signal.shape[1] // 2], wavelet_type)
    return transform

# Inverse wavelet transform. Assumes the signal length is a power of 2. Performs the inverse transform for each row.
def inv_wavelet_transform_1d(transform, wavelet_type='haar'):
    if transform.shape[1] <= 1:
        return transform
    if transform.shape[1] % 2 != 0:
        print('An error occurred in wavelet transform. Size not a power of 2.')
    transform[:, 0:(transform.shape[1] // 2)] = inv_wavelet_transform_1d(transform[:, 0:(transform.shape[1] // 2)], wavelet_type)
    signal = np.zeros_like(transform)
    principal_filter = principal_filters[wavelet_type]
    second_filter = second_filters[wavelet_type]
    for i in range(signal.shape[1]):
        for j in range(principal_filter.size // 2):
            index = (((i // 2) - j)%(transform.shape[1] // 2) + (transform.shape[1] // 2)) % (transform.shape[1] // 2)
            signal[:, i] += principal_filter[(i%2) + 2*j] * transform[:, index]
            signal[:, i] += second_filter[(i%2) + 2*j] * transform[:, index + (transform.shape[1] // 2)]
    return signal

# Wavelet transform in 2D. Assumes both image dimensions are a power of 2.
def wavelet_transform_2d(image, wavelet_type='haar'):
    image = wavelet_transform_1d(image, wavelet_type)
    image = np.transpose(wavelet_transform_1d(np.transpose(image), wavelet_type))
    return image

# Inverse wavelet transform in 2D. Assumes both image dimension are a power of 2.
def inv_wavelet_transform_2d(transform, wavelet_type='haar'):
    transform = np.transpose(inv_wavelet_transform_1d(np.transpose(transform), wavelet_type))
    transform = inv_wavelet_transform_1d(transform, wavelet_type)
    return transform

# Calculate the l1 norm of the wavelet coefficients.
def wavelet_l1(image, wavelet_type='haar'):
    transform = wavelet_transform_2d(image, wavelet_type)
    return np.sum(np.abs(transform))

# Calculate the wavelet shrink.
def wavelet_l1_prox(image, wavelet_type='haar', lam=1.0):
    transform = wavelet_transform_2d(image, wavelet_type)
    transform = np.multiply(np.sign(transform), np.multiply((np.absolute(transform) - lam > 0), np.absolute(transform) - lam))
    return inv_wavelet_transform_2d(transform, wavelet_type)

# Tests the wavelet transforms.
def main():
    array = rand(512, 512)
    transform = wavelet_transform_2d(array)
    print(np.linalg.norm(array))
    print(np.linalg.norm(transform))
    print(np.linalg.norm(inv_wavelet_transform_2d(transform) - array))
    print(wavelet_l1_prox(array, lam=0.1))

if __name__ == '__main__':
    main()
