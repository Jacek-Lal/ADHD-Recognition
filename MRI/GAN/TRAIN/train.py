import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

from MRI.config import *
from MRI.mri_filter import *
from MRI.mri_plot import *
from MRI.mri_read import *


optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

def generator(latent_dim):

    model = Sequential()

    # Hidden Layer 1: Start with 15 x 15 image
    n_nodes = 15 * 15 * 128
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(Reshape((15, 15, 128)))

    # Hidden Layer 2: Upsample to 30 x 30
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(ReLU())

    # Hidden Layer 3: Upsample to 60 x 60
    model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(ReLU())

    # Hidden Layer 4: Upsample to 120 x 120
    model.add(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(ReLU())

    # Output Layer (Grayscale would have only 1 filter)
    model.add(Conv2D(filters=1, kernel_size=(5, 5), activation='tanh', padding='same'))
    return model


def discriminator(in_shape):

    model = Sequential()

    # Hidden Layer 1
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))

    # Hidden Layer 2
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))

    # Hidden Layer 3
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))

    # Flatten and Output Layers
    model.add(Flatten())
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def def_gan(generator, discriminator):

    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


def real_samples(n, dataset):

    X = dataset[np.random.choice(dataset.shape[0], n, replace=True), :]
    y = np.ones((n, 1))

    return X, y

def latent_vector(batch_size, latent_dim):

    latent_input = np.random.randn(batch_size * latent_dim)
    latent_input = latent_input.reshape(batch_size, latent_dim)

    return latent_input

def fake_samples(generator, latent_dim, n):

    latent_output = latent_vector(latent_dim, n)

    X = generator.predict(latent_output)
    y = np.zeros((n, 1))

    return X, y

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch):

    # Our batch to train the discriminator will consist of half real images and half fake (generated) images
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):

        # Discriminator training

        x_real, y_real = real_samples(half_batch, dataset)
        x_fake, y_fake = fake_samples(g_model, latent_dim, half_batch)

        # Train the discriminator using real and fake samples
        X, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
        discriminator_loss, discriminator_acc = d_model.train_on_batch(X, y)

        # Generator training
        x_gan = latent_vector(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))

        # Train the generator via a composite GAN model
        generator_loss = gan_model.train_on_batch(x_gan, y_gan)

        # Evaluate the model at every n_eval epochs
        if (i) % 500 == 0:
            sample_noise = latent_vector(1, latent_dim)
            generated_sample = g_model.predict(sample_noise)
            plt.imshow(generated_sample[0])
            plt.title(f"Epoch: {i}, D loss: {discriminator_loss:}, A acc: {discriminator_acc:}, G loss: {generator_loss:}")
            plt.show()

    return gan_model, discriminator_acc


def train_GAN(save, data_type):

    if data_type == "ADHD":
        data = readPickle(PICKLE_DATA_ADHD_PATH)

    elif data_type == "CONTROL":
        data = readPickle(PICKLE_DATA_CONTROL_PATH)

    else:
        print("data_type ADHD LUB CONTROL")
        return

    X_train = np.array(trim(data))

    X_train = normalize(X_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

    gen_model = generator(latent_dim)

    dis_model = discriminator(in_shape=image_dim)

    gan_model = def_gan(gen_model, dis_model)

    gen, d_acc = train(gen_model, dis_model, gan_model, X_train, latent_dim, n_epochs=epochs, n_batch=batch_size)

    if save == True:
        if data_type == "ADHD":
            gen.save(f"{GAN_MODELS_PATH}/ADHD_{round(d_acc, 4)}.h5")
        elif data_type == "CONTROL":
            gen.save(f"{GAN_MODELS_PATH}/CONTROL_{round(d_acc, 4)}.h5")