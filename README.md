# Math555 Optimization Theory Materials

Here we provide some code for the Math555 Optimization Theory course at Penn State
The examples we provide are applications of optimization to image processing.

## Wavelet Compression Example

In this experiment, we demonstrate the ability of wavelets to compress natural images.
We calculate the wavelet coefficients of an image and only keep a very small fraction of
the largest coefficients. This suffices to reconstruct an image which is visually nearly
identical to the original image. A compression ratio is calculated based upon the number
of non-zero coefficients and the amount of space it takes to store their location. This
technique is very powerful, simple images can be compressed by a factor of almost 100.
For play with this experiment, simply run compression_example.py.

## Wavelet Denoising Example

The goal of this experiment is to test an image denoising algorithm based
on wavelets. Wavelets are an orthonormal basis in which most natural images
are sparse, as shown in the previous example. We will use this phenomenon to
denoise images which have been corrupted by random noise. A common way to 
do this is to solve the following optimization problem

argmin_f (1/2)|f - I|^2 + lambda*|Qf|_1,

where lambda is a parameter which depends upon the amount of noise in the image
and QF denotes the wavelet transform. To test this, run the code in
denoising_example.py.
