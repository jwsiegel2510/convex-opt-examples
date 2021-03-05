# Author: Jonathan Siegel
#
# Contains a rudimentary inpainting and denoising example to play around with.

from utils import load_image, plot_image
import numpy as np

def main():
    image = load_image()
    plot_image(image)

if __name__ == '__main__':
    main()
