from nilearn import plotting
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def plot_mri(img):
    for i in range(img.shape[-1]):
        volume_to_visualize = img.slicer[..., i]
        colors = np.ones(volume_to_visualize.shape, dtype=bool)

        plotting.plot_epi(volume_to_visualize)
        plotting.show()
        plt.close()

# Display the plot