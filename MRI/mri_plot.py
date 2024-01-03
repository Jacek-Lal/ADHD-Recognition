from nilearn import plotting
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def plot_mri(img):
    plt.imshow(img)
    plt.show()
