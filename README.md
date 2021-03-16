# Math555 Optimization Theory Materials

Here we provide code for the Math555 Optimization Theory course at Penn State

## Wavelet Denoising Example

The goal of the first problem set is to implement an image denoising algorithm
using wavelets. Wavelets are an orthonormal basis in which most natural images
are sparse (i.e. only a small number of wavelet coefficients suffices to
reconstruct an image which is visually nearly identical to the original).
You can observe this phenomenon by running compression_example.py, which
compresses an image by retaining only a fraction of its wavelet coefficients.

The goal for this problem is to use the wavelet transform to denoise a noisy image.
Given the sparsity of the wavelet basis, a common way to do this is to solve the 
following optimization problem

argmin_f (1/2)|f - I|^2 + lambda*|Qf|_1,

where lambda is a parameter which depends upon the amount of noise in the image.
In your assignment, you were asked to give the solution to the above problem
in terms of the wavelet transform Q and inverse wavelet transform Q^{-1}.

Both the wavelet transform and inverse transform have been implemented for you, so you simply have
to implement a function which solves the above problem. This function should be implemented in 
algorithms/wavelet_transform.py. Afterward, you should be able to run 
denoising_example.py.
