# Author: Jonathan Siegel
#
# For comparison, implements a trivial algorithm for reconstructing the image.

import numpy as np

def trivial_reconstruct(corr_image, mask):
    rec_image = corr_image
    while np.sum(mask) < np.sum(np.ones_like(mask)) - 0.5:
        for i in range(rec_image.shape[0]):
            for j in range(rec_image.shape[1]):
                if mask[i][j] < 0.5:
                    nums = 0
                    for k in [-1,0,1]:
                        for l in [-1,0,1]:
                            if i+k >= 0 and i+k < rec_image.shape[0] and j+l >= 0 and j+l < rec_image.shape[1] and mask[i+k][j+l] > 0.5:
                                nums += 1
                                rec_image[i][j] += rec_image[i+k][j+l]
                    if nums > 0:
                        rec_image[i][j] /= nums
                        mask[i][j] = 1.0
    return rec_image
