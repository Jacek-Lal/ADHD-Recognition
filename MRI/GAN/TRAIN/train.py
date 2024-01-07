import os
import sys
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, models, optimizers
from keras.datasets import mnist
from MRI.config import *
from MRI.mri_filter import *
from MRI.mri_plot import *


# Tensorflow / Keras
from tensorflow import keras # for building Neural Networks

from keras.models import Sequential # for assembling a Neural Network model
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Dropout # adding layers to the Neural Network model
from tensorflow.keras.utils import plot_model # for plotting model diagram
from tensorflow.keras.optimizers import Adam # for model optimization


# Data manipulation
import numpy as np # for data manipulation

import sklearn

from sklearn.preprocessing import MinMaxScaler # for scaling inputs used in the generator and discriminator


# Visualization


import matplotlib
import matplotlib.pyplot as plt # or data visualizationa





# Other utilities
import sys
import os

# Assign main directory to a variable
main_dir=os.path.dirname(sys.path[0])
#print(main_dir)

def trim(data, nr_rows=4):
    trimmed = copy.deepcopy(data)

    for i in range(len(data)):
        trimmed[i] = data[i][nr_rows:-nr_rows]

    return trimmed


import numpy as np
from PIL import Image

with open(r"/home/user/Desktop/ADHD-Recognition/MRI/PICKLE_DATA/controlImages.pkl", 'rb') as file:
    data = pickle.load(file)

X_train = np.array(trim(data))

X_train = normalize(X_train)
# X_train__ = []
# nowy_rozmiar = (64, 64)
#
# for i in range(X_train.shape[0]):
#     image_pillow = Image.fromarray((X_train[i] * 255).astype(np.uint8))
#     zmieniony_obraz_pillow = image_pillow.resize(nowy_rozmiar)
#     zmieniony_obraz = np.array(zmieniony_obraz_pillow) / 255.0
#     X_train__.append(zmieniony_obraz)


def generate_noise(batch_size, noise_dim):
    x_input = np.random.randn(batch_size * noise_dim)
    x_input = x_input.reshape(batch_size, noise_dim)
    # np.random.normal(0, 1, size=(batch_size, noise_dim))

    return x_input


# X_train = np.array(X_train__)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

plot_mri(X_train[0])

def generator(latent_dim):
    model = Sequential(name="Generator")  # Model

    # Hidden Layer 1: Start with 8 x 8 image
    n_nodes = 5 * 5 * 128  # number of nodes in the first hidden layer
    model.add(Dense(n_nodes, input_dim=latent_dim, name='Generator-Hidden-Layer-1'))
    model.add(Reshape((5, 5, 128), name='Generator-Hidden-Layer-Reshape-1'))

    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(3, 3), padding='same',
                              name='Generator-Hidden-Layer-1.2'))
    model.add(ReLU(name='Generator-Hidden-Layer-Activation-1.2'))
    # Hidden Layer 2: Upsample to 16 x 16
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                              name='Generator-Hidden-Layer-2'))
    model.add(ReLU(name='Generator-Hidden-Layer-Activation-2'))

    # Hidden Layer 3: Upsample to 32 x 32
    model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                              name='Generator-Hidden-Layer-3'))
    model.add(ReLU(name='Generator-Hidden-Layer-Activation-3'))

    # Hidden Layer 4: Upsample to 64 x 64
    model.add(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same',
                              name='Generator-Hidden-Layer-4'))
    model.add(ReLU(name='Generator-Hidden-Layer-Activation-4'))

    # Output Layer (Note, we use 3 filters because we have 3 channels for a color image. Grayscale would have only 1 channel)
    model.add(Conv2D(filters=1, kernel_size=(5, 5), activation='tanh', padding='same', name='Generator-Output-Layer'))
    return model

# Instantiate
latent_dim=100 # Our latent space has 100 dimensions. We can change it to any number
gen_model = generator(latent_dim)


def discriminator(in_shape=(120, 120, 1)):
    model = Sequential(name="Discriminator")  # Model

    # Hidden Layer 1
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                     name='Discriminator-Hidden-Layer-1'))
    model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1'))

    # Hidden Layer 2
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                     name='Discriminator-Hidden-Layer-2'))
    model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2'))

    # Hidden Layer 3
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                     name='Discriminator-Hidden-Layer-3'))
    model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-3'))

    # Flatten and Output Layers
    model.add(Flatten(name='Discriminator-Flatten-Layer'))  # Flatten the shape
    model.add(Dropout(0.3,
                      name='Discriminator-Flatten-Layer-Dropout'))  # Randomly drop some connections for better generalization
    model.add(Dense(1, activation='sigmoid', name='Discriminator-Output-Layer'))  # Output Layer

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model


# Instantiate
dis_model = discriminator()


def def_gan(generator, discriminator):
    # We don't want to train the weights of discriminator at this stage. Hence, make it not trainable
    discriminator.trainable = False

    # Combine
    model = Sequential(name="DCGAN")  # GAN Model
    model.add(generator)  # Add Generator
    model.add(discriminator)  # Add Disriminator

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model


gan_model = def_gan(gen_model, dis_model)


def real_samples(n, dataset):
    # Samples of real data
    X = dataset[np.random.choice(dataset.shape[0], n, replace=True), :]

    # Class labels
    y = np.ones((n, 1))

    return X, y


def latent_vector(latent_dim, n):
    # Generate points in the latent space
    latent_input = np.random.randn(latent_dim * n)

    # Reshape into a batch of inputs for the network
    latent_input = latent_input.reshape(n, latent_dim)
    return latent_input


def fake_samples(generator, latent_dim, n):
    # Generate points in latent space
    latent_output = latent_vector(latent_dim, n)

    # Predict outputs (i.e., generate fake samples)
    X = generator.predict(latent_output)

    # Create class labels
    y = np.zeros((n, 1))
    return X, y


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10000, n_batch=32, n_eval=500):
    # Our batch to train the discriminator will consist of half real images and half fake (generated) images
    half_batch = int(n_batch / 2)

    # We will manually enumare epochs
    for i in range(n_epochs):

        # Discriminator training
        # Prep real samples
        x_real, y_real = real_samples(half_batch, dataset)
        # Prep fake (generated) samples
        x_fake, y_fake = fake_samples(g_model, latent_dim, half_batch)

        # Train the discriminator using real and fake samples
        X, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
        discriminator_loss, _ = d_model.train_on_batch(X, y)

        # Generator training
        # Get values from the latent space to be used as inputs for the generator
        x_gan = latent_vector(latent_dim, n_batch)
        # While we are generating fake samples,
        # we want GAN generator model to create examples that resemble the real ones,
        # hence we want to pass labels corresponding to real samples, i.e. y=1, not 0.
        y_gan = np.ones((n_batch, 1))

        # Train the generator via a composite GAN model
        generator_loss = gan_model.train_on_batch(x_gan, y_gan)

        # Evaluate the model at every n_eval epochs
        if (i) % n_eval == 0:
            # Generowanie przykładowego obrazka po zakończeniu treningu
            sample_noise = generate_noise(1, noise_dim)
            generated_sample = g_model.predict(sample_noise)
            plot_mri(generated_sample[0])



train(gen_model, dis_model, gan_model, X_train, latent_dim)
