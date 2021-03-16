# Math555 Optimization Theory Materials

Here we provide code for the Math555 Optimization Theory course at Penn State

## Wavelet Denoising Example

The goal of the first problem set is to implement an image denoising algorithm
using wavelets. The wavelet transform has been implemented for you, you simple have
to implement a function which solves the proximal map

argmin_f (1/2)|f - I|^2 + |Qf|_1

where Q is the wavelet transform. This function should be implemented in 
algorithms/wavelet_transform.py. Afterward, you should be able to run the 
denoising example code.
