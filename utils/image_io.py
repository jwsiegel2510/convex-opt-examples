# Author: Jonathan Siegel
#
# Contains basic input and output routines for images.

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

# Read an image and normalize to [0,1].
def load_image(filename='lenna-image.jpeg'):
    image = imread(filename)
    return image / np.max(image)

# Plot an image.
def plot_image(img):
    plt.imshow(img, cmap='Greys_r')
    plt.show()
