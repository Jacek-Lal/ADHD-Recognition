from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from MRI.config import *

def showPhoto(X):

    plt.imshow((X*-1).reshape(28, 28), cmap="gray")

    plt.show()

def generate_noise(batch_size, noise_dim):
    x_input = np.random.randn(batch_size * noise_dim)

    x_input = x_input.reshape(batch_size, noise_dim)

    #np.random.normal(0, 1, size=(batch_size, noise_dim))

    return x_input


MODEL_NAME = "12345"

generator = load_model(f"{GAN_MODEL_PATH}/{MODEL_NAME}.h5")

sample_noise = generate_noise(1, noise_dim)

generated_sample = generator.predict(sample_noise)

showPhoto(generated_sample)
