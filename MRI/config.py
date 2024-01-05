TASKS = ['SLD','SLI','SSD','SSI','VLD','VLI','VSD','VSI']
MRI_DATA_PATH = r"MRI\files"
PATIENTS_DATA_PATH = r"MRI\files\participants.tsv"
GAN_MODEL_PATH = "../MODEL/"

# Parametry
noise_dim = 100 # Wymiar szumu generowanego przez generator.
image_dim = (120, 120, 1)  # Rozmiar obrazu
batch_size = 16 # Liczba obrazów w jednej iteracji treningowej.
epochs = 10000 # Liczba epok treningowych.