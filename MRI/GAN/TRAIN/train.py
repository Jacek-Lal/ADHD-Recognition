import os
import sys
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pickle
import numpy as np
from tensorflow import keras 
from keras import layers, models

from MRI.mri_filter import *
from MRI.config import *
from MRI.mri_read import *
from MRI.mri_plot import *

data = readPickle(PICKLE_DATA_ADHD_PATH)
 
X_train = np.array(trim(data))

def generate_noise(batch_size, noise_dim):

    x_input = np.random.randn(batch_size * noise_dim)
    x_input = x_input.reshape(batch_size, noise_dim)
    #np.random.normal(0, 1, size=(batch_size, noise_dim))

    return x_input

def build_generator(noise_dim, output_dim):

    model = models.Sequential()

    model.add(layers.Dense(128, input_dim=noise_dim, activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(np.prod(output_dim), activation='sigmoid'))
    model.add(layers.Reshape(output_dim))

    return model

def build_discriminator(input_dim):

    model = models.Sequential()

    model.add(layers.Flatten(input_shape=input_dim))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def build_gan(generator, discriminator):

    discriminator.trainable = False
    model = models.Sequential()

    model.add(generator)
    model.add(discriminator)

    return model


discriminator = build_discriminator(image_dim)
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

generator = build_generator(noise_dim, image_dim)
generator.compile(optimizer='adam', loss='binary_crossentropy')

gan = build_gan(generator, discriminator)
gan.compile(optimizer='adam', loss='binary_crossentropy')


for epoch in range(epochs):
    
    # Pobranie rzeczywistych obrazów
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    labels_real = np.ones((batch_size, 1))

    #  Generowanie obrazów przy użyciu generatora
    noise = generate_noise(batch_size, noise_dim)
    generated_images = generator.predict(noise)
    labels_fake = np.zeros((batch_size, 1))

    # Trening dyskryminatora na rzeczywistych i wygenerowanych danych
    d_loss_real, d_acc_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(generated_images, labels_fake)

    # Obliczenie łącznej straty i dokładności dla dyskryminatora
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)

    # Generowanie niby prawidlowego zdjecia
    noise = generate_noise(batch_size, noise_dim)
    labels_gan = np.ones((batch_size, 1))

    # Trening generatora na wygenerowanym szumie
    g_loss = gan.train_on_batch(noise, labels_gan)

    if epoch % 100 == 0:
        # Wydruk wartości straty dla dyskryminatora i generatora
        print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

        # Generowanie przykładowego obrazka po zakończeniu treningu
        sample_noise = generate_noise(1, noise_dim)
        generated_sample = generator.predict(sample_noise)
        plot_mri(generated_sample[0])

generator.save(f"{round(g_loss, 4)}.h5")