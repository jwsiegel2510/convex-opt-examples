# Author: Jonathan Siegel
#
# Contains basic input and output routines for images.

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

# Read an image and normalize to [0,1].
def load_image(filename='mickey.jpg'):
    image = imread(filename)
    if len(image.shape) == 3:
        image = np.sum(image, axis=-1)
    return image / np.max(image)

# Plot an image.
def plot_image(img):
    plt.imshow(img, cmap='Greys_r')
    plt.show()
