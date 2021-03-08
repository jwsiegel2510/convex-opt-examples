# Author: Jonathan Siegel
#
# Reconstructs an image which has had many pixels removed using an optimization of the form
# 
# min_f (1/2)*\|f - I\|^2_{mask} + lam*|Wf|_1,
#
# where the first terms is an l2 loss on the known pixel values and the second term is the l1 norm
# of the wavelet transform of Wf of f.
#
# If the algorithm is set to dr (i.e. Douglas Rachford), then this corresponds to
# replacing the l2 norm by a constraint, i.e. setting lambda = infty. The value for
# lambda then effectively acts as a step size.

import numpy as np
from .wavelet_transform import wavelet_l1, wavelet_l1_prox

def reconstruct(corr_img, mask, lam=0.1, wavelet_order=0, alg='fb', max_iter=100):
    # Initialize the image.
    rec_img = corr_img
    # Implement the forward-backward algorithm.
    if alg == 'fb':
        for it_count in range(max_iter):
            # Perform the forward step (with step size 1.0 from the smoothness of the l2 term).
            rec_img = np.multiply((1.0 - mask), rec_img) + np.multiply(mask, corr_img)
            # Perform the backward step.
            rec_img = wavelet_l1_prox(rec_img, lam=lam, wavelet_order=wavelet_order)
            # Calculate and report objective.
            obs_error = np.multiply(rec_img - corr_img, mask)
            objective = 0.5 * np.linalg.norm(obs_error, ord=2)**2 + lam * wavelet_l1(rec_img, wavelet_order=wavelet_order)
            print('The objective error is: %f' % (objective))
        return np.clip(rec_img, 0, 1)
    # Implement Douglas Rachford splitting.
    if alg == 'dr':
        temp_img = rec_img
        for it_count in range(max_iter):
            ttemp_img = temp_img
            # Perform the reflected prox for the l1 term.
            temp_img = 2 * wavelet_l1_prox(temp_img, lam=lam, wavelet_order=wavelet_order) - temp_img
            # Set the recovered image to the mid point.
            rec_img = np.multiply((1.0 - mask), temp_img) + np.multiply(mask, corr_img)
            # Perform the reflected prox for the constraint.
            temp_img = 2 * (np.multiply((1.0 - mask), temp_img) + np.multiply(mask, corr_img)) - temp_img
            # Average with the identity.
            temp_img = 0.5 * temp_img + 0.5 * ttemp_img
            # Calculate and report objective.
            objective = wavelet_l1(rec_img, wavelet_order=wavelet_order)
            print('The objective error is: %f' % (objective))
        return np.clip(rec_img, 0, 1)



