import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D, ReLU, LeakyReLU, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from MRI.config import *
from MRI.mri_filter import *
from MRI.mri_read import *


def generator(latent_dim):
    model = Sequential()

    # Hidden Layer 1: Start with 5 x 5 image
    n_nodes = 15 * 15 * 128
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(Reshape((15, 15, 128)))

    # Upsample to 30x30
    model.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(ReLU())

    # Hidden Layer 2: Upsample to 60 x 60
    model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(ReLU())

    # Hidden Layer 3: Upsample to 120 x 120
    model.add(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(ReLU())

    # Output Layer for grayscale would have only 1 channel
    model.add(Conv2D(filters=1, kernel_size=(5, 5), activation='tanh', padding='same'))
    return model


latent_dim=100
gen_model = generator(latent_dim)


def discriminator(in_shape=(120, 120, 1)):

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

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))  # Output Layer

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model


# Instantiate
dis_model = discriminator()


def def_gan(generator, discriminator):
    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model


gan_model = def_gan(gen_model, dis_model)


def real_samples(n, dataset):
    X = dataset[np.random.choice(dataset.shape[0], n, replace=True), :]

    # Class labels
    y = np.ones((n, 1))

    return X, y


def latent_vector(latent_dim, n):
    latent_input = np.random.randn(latent_dim * n)

    latent_input = latent_input.reshape(n, latent_dim)
    return latent_input


def fake_samples(generator, latent_dim, n):
    latent_output = latent_vector(latent_dim, n)

    X = generator.predict(latent_output)

    y = np.zeros((n, 1))
    return X, y


def train_GAN(save, data_type, n_epochs=2000, n_batch=32, g_model=gen_model, d_model=dis_model, gan_model=gan_model, latent_dim=latent_dim):
    # SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
    if data_type == "ADHD":
        data = readPickle(f'../MRI/PICKLE_DATA/controlImages.pkl')

    elif data_type == "CONTROL":
        data = readPickle(f'../MRI/PICKLE_DATA/controlImages.pkl')

    else:
        print("data_type ADHD LUB CONTROL")
        return

    X_train = np.array(trim(data))

    X_train = normalize(X_train)

    dataset = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

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


        if (i) % 500 == 0:
            sample_noise = latent_vector(latent_dim, 1)
            generated_sample = g_model.predict(sample_noise)
            plt.imshow(generated_sample[0])
            plt.title(f"Epoch: {i}, D acc: {discriminator_acc} D loss: {discriminator_loss:.5f}, G loss: {generator_loss:.5f}")
            plt.show()

    if save == True:
        if data_type == "ADHD":
            # SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
            g_model.save(f'../MODEL/ADHD_{round(discriminator_loss, 4)}.h5')
        elif data_type == "CONTROL":
            g_model.save(f'../MODEL/CONTROL_{round(discriminator_loss, 4)}.h5')