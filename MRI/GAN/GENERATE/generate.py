from keras.models import load_model
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.abspath(__file__))

from MRI.config import *
from MRI.mri_plot import *
from MRI.GAN.TRAIN.train import latent_vector

def generate_GAN(MODEL_GAN_NAME):

    generator = load_model(f"{GAN_MODELS_PATH}/{round(MODEL_GAN_NAME, 4)}.h5")

    sample_noise = latent_vector(1, latent_dim)

    generated_sample = generator.predict(sample_noise)

    plot_mri(generated_sample[0])