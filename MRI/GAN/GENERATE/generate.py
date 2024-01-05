from keras.models import load_model
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.abspath(__file__))
from MRI.config import *

def showPhoto(X):

    plt.imshow(X, cmap="gray")

    plt.show()

def generate_noise(batch_size, noise_dim):
    x_input = np.random.randn(batch_size * noise_dim)

    x_input = x_input.reshape(batch_size, noise_dim)

    #np.random.normal(0, 1, size=(batch_size, noise_dim))

    return x_input


MODEL_NAME = "0.9531"

generator = load_model(f"{MODEL_NAME}.h5")

sample_noise = generate_noise(1, noise_dim)

generated_sample = generator.predict(sample_noise)

showPhoto(generated_sample)
