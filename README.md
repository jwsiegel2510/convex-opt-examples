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
To play with this experiment, simply run compression_example.py.

## Wavelet Denoising Example

The goal of this experiment is to test an image denoising algorithm based
on wavelets. Wavelets are an orthonormal basis in which most natural images
are sparse, as shown in the previous example. We will use this phenomenon to
denoise images which have been corrupted by random noise. A common way to 
do this is to solve the following optimization problem

argmin_f (1/2)|f - I|^2 + lambda*|Qf|_1,

where lambda is a parameter which depends upon the amount of noise in the image
and QF denotes the wavelet transform. To test this approach for denoising images with
random Gaussian noise, run denoising_example.py.

## Compressed Sensing Example

In this experiment, we implement an algorithm which reconstructs a sparse vector given a
small sample of its Fourier transform, utilizing the famous Dantzig estimator from compressed
sensing. Specifically, we generate a random vector x_0 for which only a small percentage of entries
are nonzero, the nonzero entries being random independent Gaussians. We then sample a percentage
of the entries Fourier transform of x_0 and solve the optimization problem

argmin_{Ax=b} |x|_1,

where the constraint Ax=b means that the Fourier transform of x should match that of x_0 in the
sampled entries. This optimization problem is then solved using the Douglas-Rachford iteration.
To test this experiment, run sensing_example.py. You should observe that the original vector
(of dimension 512) can be reliably recovered from about 10% of its Fourier transform as long as
only 1% of its entries are nonzero.
