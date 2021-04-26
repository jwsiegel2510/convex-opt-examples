# Author: Jonathan Siegel
#
# Implements algorithms for optimizing the total variation energy for image
# denoising. The continuous problem is discretized using piecewise constant
# function for the image and a piecewise bilinear vector field for the dual
# variable.
# 
# The algorithm used to solve this problem is accelerated forward-backward
# splitting applied to the dual problem.

import math
import copy
import numpy as np
from scipy.signal import convolve2d

# Applies the discretized gradient operator to an image. The output will be a vector field
# vanishing on the boundary.
def grad_operator(image):
    shape = image.shape
    x_grad = shape[0] * convolve2d(image, [[-0.5, -0.5], [0.5, 0.5]], mode='valid')
    y_grad = shape[1] * convolve2d(image, [[-0.5, 0.5], [-0.5, 0.5]], mode='valid')
    return (x_grad, y_grad)

# Applies the discretized divergence operator to a vector field which vanishes on the boundary.
def transpose_grad_operator(grad):
    x_grad = grad[0]
    y_grad = grad[1]
    image = np.zeros((x_grad.shape[0]+1, x_grad.shape[1]+1))
    image += image.shape[0] * convolve2d(x_grad, np.array([[0.5, 0.5], [-0.5, -0.5]]), mode='full')
    image += image.shape[1] * convolve2d(y_grad, np.array([[0.5, -0.5], [0.5, -0.5]]), mode='full')
    return image

# The proximal map is a projection onto the l-infty ball of radius lambda.
def l1_proximal(grad, lam):
    if lam == 0:
        return (np.zeros_like(grad[0]), np.zeros_like(grad[1]))
    x_grad = grad[0]
    y_grad = grad[1]
    norms = x_grad ** 2 + y_grad ** 2
    norms = np.sqrt(norms)
    x_grad = lam / np.clip(norms, a_min=lam, a_max=None) * x_grad
    y_grad = lam / np.clip(norms, a_min=lam, a_max=None) * y_grad
    return (x_grad, y_grad)

# Evaluate l2 objective.
def grad_l2_objective(grad, noisy_image):
    error = transpose_grad_operator(grad) - noisy_image
    return (1/2) * np.inner(error, error)

# Denoise by minimizing the total variation functional.
def total_variation_denoising(noisy_image, lam=1.0, num_iter=200):
    # Initialize the dual gradient variable and other iterates.
    grad = (np.zeros_like(grad_operator(noisy_image)[0]), np.zeros_like(grad_operator(noisy_image)[1]))
    grad_temp = copy.deepcopy(grad)
    # Determine appropriate step size.
    maximum_shape = max(noisy_image.shape[0], noisy_image.shape[1])
    step = 1.0 / (8 * maximum_shape ** 2)
    # Set parameters for momentum step.
    q = 0
    qp1 = math.sqrt(5) - 1
    for i in range(num_iter):
        # Evaluate gradient at grad_temp.
        gradient = grad_operator(transpose_grad_operator(grad_temp) - noisy_image)
        # Take a forward-backward step.
        grad_new = l1_proximal((grad_temp[0] - step * gradient[0], grad_temp[1] - step * gradient[1]), lam)
        # Perform extrapolation.
        alpha = q / (2 + qp1)
        grad_temp = (grad_new[0] + alpha * (grad_new[0] - grad[0]), grad_new[1] + alpha * (grad_new[1] - grad[1]))
        grad = copy.deepcopy(grad_new)
        q = qp1
        qp1 = math.sqrt((q + 2) ** 2 + 1) - 1
    # Recover image from dual iterate.
    return noisy_image - transpose_grad_operator(grad)
    

# Routine which tests parts of the implementation.
def main():
    image = np.random.random(size=(3,3))
    print(image)
    grad = grad_operator(image)
    print(grad)
    print(transpose_grad_operator(grad))
    print(l1_proximal(grad, 0.1))

if __name__ == '__main__':
    main()
