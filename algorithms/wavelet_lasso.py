# Author: Jonathan Siegel
#
# Reconstructs an image which has had many pixels removed using an optimization of the form
# 
# min_f (1/2)*\|f - I\|^2_{mask} + lam*|Wf|_1,
#
# where the first terms is an l2 loss on the known pixel values and the second term is the l1 norm
# of the wavelet transform of Wf of f.

import numpy as np
from .wavelet_transform import wavelet_l1, wavelet_l1_prox

def reconstruct(corr_img, mask, lam=0.1, wavelet_type='haar', alg='fb', max_iter=100):
    # Initialize the image.
    rec_img = corr_img
    # Implement the forward-backward algorithm.
    if alg == 'fb':
        for it_count in range(max_iter):
            # Perform the forward step (with step size 1.0 from the smoothness of the l2 term).
            rec_img = np.multiply((1.0 - mask), rec_img) + np.multiply(mask, corr_img)
            # Perform the backward step.
            rec_img = wavelet_l1_prox(rec_img, lam=lam)
            # Calculate and report objective.
            obs_error = np.multiply(rec_img - corr_img, mask)
            objective = 0.5 * np.linalg.norm(obs_error, ord=2)**2 + lam * wavelet_l1(rec_img)
            print('The objective error is: %f' % (objective))
        return np.clip(rec_img, 0, 1)
