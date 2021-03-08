# Author: Jonathan Siegel
#
# Reconstructs an image from a sample of its Fourier transform using an optimization of the form
# 
# min_f |Wf|_1 (the Danzig estimator),
#
# with the constraint that the transform of f must match the samples. Here the objective is the l1 norm
# of the wavelet transform of Wf of f, or 
#
# min_f (1/2)|Ff - S|^2 + lam|Wf|_1 (lasso estimator),
#
# where the l2 norm penalizes the difference between the transform of f and the samples.
#
# The algorithm used to solve the Danzig problem is the Douglas-Rachford algorithm.
# 
# The algorithm used to solve the lasso is forward-backward descent.

import numpy as np
from .wavelet_transform import wavelet_l1, wavelet_l1_prox

# Project onto the constraint set.
def fft_projection(img, sampled_transform, mask):
    transform = np.fft.rfftn(img, norm='ortho')
    new_transform = np.multiply((1.0 - mask), transform) + np.multiply(mask, sampled_transform)
    return np.fft.irfftn(new_transform, norm='ortho')

# Calculate the fft l2 error.
def fft_error(img, sampled_transform, mask):
    transform = np.fft.rfftn(img, norm='ortho')
    return np.linalg.norm(np.multiply(mask, transform - sampled_transform))

def reconstruct_danzig(sampled_transform, mask, step=0.1, wavelet_order=0, max_iter=100):
    # Initialize the image.
    rec_img = np.fft.irfftn(sampled_transform, norm='ortho')
    # Calculate and report objective.
    objective = wavelet_l1(rec_img, wavelet_order=wavelet_order)
    print('The objective error is: %f' % (objective))
    # Implement Douglas Rachford splitting.
    temp_img = rec_img
    for it_count in range(max_iter):
        old_temp_img = temp_img
        # Perform the reflected prox for the l1 term.
        temp_img = 2 * wavelet_l1_prox(temp_img, lam=step, wavelet_order=wavelet_order) - temp_img
        # Set the recovered image to the mid point.
        rec_img = fft_projection(temp_img, sampled_transform, mask)
        # Perform the reflected prox for the constraint.
        temp_img = 2 * rec_img - temp_img
        # Average with the identity.
        temp_img = 0.5 * temp_img + 0.5 * old_temp_img
        # Calculate and report objective.
        objective = wavelet_l1(rec_img, wavelet_order=wavelet_order)
        print('The objective error is: %f' % (objective))
    return rec_img

def reconstruct_lasso(sampled_transform, mask, lam=0.1, wavelet_order=0, max_iter=100):
    # Initialize the image.
    rec_img = np.fft.irfftn(sampled_transform, norm='ortho')
    # Calculate and report objective.
    objective = 0.5 * fft_error(rec_img, sampled_transform, mask) ** 2 + lam * wavelet_l1(rec_img, wavelet_order=wavelet_order)
    print('The objective error is: %f' % (objective))
    # Implement Forward Backward splitting.
    for it_count in range(max_iter):
        # Perform forward step with step size 1.0 corresponding to the smoothness of the l2 norm part of the objective.
        rec_img = fft_projection(rec_img, sampled_transform, mask)
        # Perform backwrad step.
        rec_img = wavelet_l1_prox(rec_img, lam=lam, wavelet_order=wavelet_order)
        # Calculate and report objective.
        objective = 0.5 * fft_error(rec_img, sampled_transform, mask) ** 2 + lam * wavelet_l1(rec_img, wavelet_order=wavelet_order)
        print('The objective error is: %f' % (objective))
    return rec_img


