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
from MRI.GAN.TRAIN.train import latent_vector

def generate_GAN(MODEL_GAN_NAME, im_amount, data_type):


    if data_type == "ADHD":
        generator = load_model(f"{GAN_MODELS_PATH}/ADHD_{round(MODEL_GAN_NAME, 4)}.h5")

    elif data_type == "CONTROL":
        generator = load_model(f"{GAN_MODELS_PATH}/CONTROL_{round(MODEL_GAN_NAME, 4)}.h5")

    else:
        print("data_type: ADHD LUB CONTROL")
        return

    data = []

    for i in range(im_amount):

        sample_noise = latent_vector(1, latent_dim)

        generated_sample = generator.predict(sample_noise)

        data.append(generated_sample[0])

    return data