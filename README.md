# Math555 Optimization Theory Materials

Here we provide some code for the Math555 Optimization Theory course at Penn State,
cotaught by Jonathan Siegel and Jinchao Xu.

The examples we provide are mostly applications of optimization to some problems in 
image processing.

## Wavelet Compression Example

In this experiment, we demonstrate the ability of wavelets to compress natural images.
We calculate the wavelet coefficients of an image and only keep a very small fraction of
the largest coefficients. This suffices to reconstruct an image which is visually nearly
identical to the original image. A compression ratio is calculated based upon the number
of non-zero coefficients and the amount of space it takes to store their location. This
technique is very powerful, simple images can be compressed by a factor of almost 100.

To play with this experiment, simply run **compression_example.py**.

## Wavelet Denoising Example

The goal of this experiment is to test an image denoising algorithm based
on wavelets. Wavelets are an orthonormal basis in which most natural images
are sparse, as shown in the previous example. We will use this phenomenon to
denoise images which have been corrupted by random noise. A common way to 
do this is to solve the following optimization problem

argmin_f (1/2)|f - I|^2 + lambda*|Qf|_1,

where lambda is a parameter which depends upon the amount of noise in the image
and QF denotes the wavelet transform. 

To test this approach for denoising images with
random Gaussian noise, run **wavelet_denoising_example.py**.

## Compressed Sensing Example

In this experiment, we implement an algorithm which reconstructs a sparse vector given a
small sample of its Fourier transform, utilizing the famous Dantzig estimator from compressed
sensing. Specifically, we generate a random vector x_0 for which only a small percentage of entries
are nonzero, the nonzero entries being random independent Gaussians. We then sample a percentage
of the entries Fourier transform of x_0 and solve the optimization problem

argmin_{Ax=b} |x|_1,

where the constraint Ax=b means that the Fourier transform of x should match that of x_0 in the
sampled entries. This optimization problem is then solved using the Douglas-Rachford iteration.

To test this experiment, run **sensing_example.py**. 

You should observe that the original vector
(of dimension 512) can be reliably recovered from about 10% of its Fourier transform as long as
only 1% of its entries are nonzero.

## Total Variation Denoising Example

In this experiment, we test the total variation method for image denoising. In this method, a noisy
image f is denoised by solving the optimization problem

argmin_u (1/2)|f - u|^2 + lambda*|Du|_1,

where D represents to gradient operator, |Du|_1 is the L1-norm of the gradient or total variation norm,
and |f-u| is the L2-norm of the error between f and u. The total variation norm can more generally be
defined for functions which are not necessarily differentiable, which leads to the more precise primal dual
formulation of the total variation objective

argmin_u max_{v:|v|<=lambda} (1/2)|f - u|^2 + (D.v,u),

where the maximum is taken over all vector fields v which vanish at the boundary of the domain and whose
magnitude does not exceed lambda. Here D.v denotes the divergence and (.,.) is the L2-inner product. 

We discretize this problem by taking the function u to be piecewise constant on a grid and taking the dual
variable v to be piecewise bilinear on the same grid and vanishing on the boundary. Then we formulate
and solve the dual problem using accelerated forward-backward splitting and reconstruct the primal
solution u. The method is quite effective, especially on the piecewise constant image mickey.jpg.

To test this method, run **total_variation_denoising_example.py**.
