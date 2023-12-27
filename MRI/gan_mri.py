import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pickle
from tensorflow.keras import layers, models
from mri_read import *

# Funkcja wczytująca obrazy z pliku .pkl
def load_images_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    images = []
    for patient_data['data'].get_fdata() in data:
        for i in range(patient_data.shape[-1]):
            images.append(patient_data[:,:,:,i])

    return np.array(images)

# Parametry
noise_dim = 100
image_dim = (128, 120, 32, 1)  # Rozmiar obrazu po dodaniu warstwy Reshape
batch_size = 64
epochs = 1500

# Wczytanie danych z pliku .pkl
data_path = r'C:\Users\Radek\Desktop\IPZ\GIT\ADHD-Recognition\MRI\lista.pkl'
X_train = load_images_from_pickle(data_path)

# Funkcja generująca losowy szum jako dane wejściowe dla generatora
def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))

# Funkcja tworząca generator
def build_generator(noise_dim, output_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=noise_dim, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(np.prod(output_dim), activation='sigmoid'))
    model.add(layers.Reshape(output_dim))  # Dodaj warstwę Reshape
    return model

# Funkcja tworząca dyskryminator
def build_discriminator(input_dim):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_dim))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Funkcja tworząca model GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Tworzenie i kompilacja modeli
discriminator = build_discriminator(image_dim)
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

generator = build_generator(noise_dim, image_dim)
generator.compile(optimizer='adam', loss='binary_crossentropy')

gan = build_gan(generator, discriminator)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Pętla ucząca
for epoch in range(epochs):
    # Trening dyskryminatora
    idx = np.random.randint(0, len(X_train), batch_size)
    real_images = np.array(X_train)[idx]
    labels_real = np.ones((batch_size, 1))

    noise = generate_noise(batch_size, noise_dim)
    generated_images = generator.predict(noise)
    labels_fake = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Trening generatora
    noise = generate_noise(batch_size, noise_dim)
    labels_gan = np.ones((batch_size, 1))

    g_loss = gan.train_on_batch(noise, labels_gan)

    # Wydruk statystyk co kilka epok
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Generowanie przykładowego obrazka po zakończeniu treningu
sample_noise = generate_noise(1, noise_dim)
generated_sample = generator.predict(sample_noise).reshape(128, 120, 32)

# Wyświetlenie przykładowego obrazka
plt.imshow(generated_sample[:, :, 16], cmap='gray')  # Wybierz warstwę 16 z wymiaru 32
plt.show()

# Zapisanie wygenerowanego obrazka w formacie NIfTI
output_path = 'generated_image.nii.gz'
nifti_image = nib.Nifti1Image(generated_sample, np.eye(4))
nib.save(nifti_image, output_path)
