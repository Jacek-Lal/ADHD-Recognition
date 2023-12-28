import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras 
from keras import layers, models
from mri_read import *

# Funkcja wczytująca obrazy z pliku .pkl
def load_images_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    images = []
    labels = []
    for patient in data:
        patientData = getData(patient['data'])
        for i in range(patientData.shape[-1]):
            images.append(patientData[:,:,:,i])
            labels.append(patient['hasAdhd'])

    return np.array(images), np.array(labels)

# Funkcja generująca losowy szum jako dane wejściowe dla generatora
def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))

# Funkcja tworząca generator
def normalize(data):
    min = np.min(data)
    max = np.max(data)
    return (data-min)/(max-min)

# Zakładając, że obrazy MRI są przechowywane w tablicy NumPy o nazwie 'mri_images'
# o wymiarach (num_samples, 128, 120, 32)

# Generator Model
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(128 * 120 * 32, input_dim=latent_dim))
    model.add(layers.Reshape((128, 120, 32, 1)))
    model.add(layers.Conv3DTranspose(128, (4, 4, 4), strides=(1, 1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv3DTranspose(64, (4, 4, 4), strides=(1, 1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv3DTranspose(1, (4, 4, 4), activation='sigmoid', padding='same'))
    model.summary()
    return model

# Discriminator Model
def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Conv3D(64, (4, 4, 4), strides=(1, 1, 1), padding='same', input_shape=[128,120,32,1]))
    
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv3D(128, (4, 4, 4), strides=(1, 1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model

# Combined Model (Generator + Discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    
    return model

# Konfiguracja GAN
latent_dim = 100
img_shape = (128, 120, 32, 1)
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

#print(generator.output_shape)
#print(discriminator.input_shape)
# Kompilacja dyskryminatora
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
discriminator.trainable = False

# Kompilacja GAN
gan = build_gan(generator, discriminator)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Trenowanie GAN
epochs = 10
batch_size = 32
X_data, y_data = load_images_from_pickle('lista.pkl')
mri_images = normalize(X_data)

for epoch in range(epochs):
    print(1)
    # Generowanie szumu z rozkładu normalnego jako wejście do generatora
    # Generowanie szumu z rozkładu normalnego jako wejście do generatora
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise)
    print(2)

    # Losowy wybór rzeczywistych obrazów z danych wejściowych
    idx = np.random.randint(0, mri_images.shape[0], batch_size)
    real_images = mri_images[idx]

    print(3)
    # Etykiety dla rzeczywistych i wygenerowanych obrazów
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))

    print(4)
    
    # Trenowanie dyskryminatora na rzeczywistych i wygenerowanych danych
    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
    print(5)

    # Obliczanie straty dyskryminatora
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    print(6)

    # Generowanie nowego szumu dla wejścia do GAN
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    labels_gan = np.ones((batch_size, 1))

    print(7)
    # Trenowanie GAN na wygenerowanych danych
    g_loss = gan.train_on_batch(noise, labels_gan)
    print(8)

    # Wydrukowanie straty co pewną liczbę epok
    print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Generowanie syntetycznych obrazów za pomocą wyuczonego generatora
synthetic_images = generator.predict(np.random.normal(0, 1, (1, latent_dim)))
