from keras.models import load_model
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.abspath(__file__))

from MRI.config import *
from MRI.mri_plot import *
from MRI.mri_read import *
from MRI.GAN.TRAIN.train import latent_vector, latent_dim


def generate_GAN(MODEL_GAN_NAME, im_amount):

    # SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
    generator = load_model(f"/home/user/Desktop/ADHD-Recognition/MRI/GAN/MODELS/{MODEL_GAN_NAME}.h5")

    data = []

    for i in range(im_amount):

        sample_noise = latent_vector(latent_dim, 1)

        generated_sample = generator.predict(sample_noise)

        data.append(generated_sample[0])

    return data