# Parametry GAN
latent_dim = 100 # Wymiar szumu generowanego przez generator.
image_dim = (120, 120, 1)
batch_size = 32 # Liczba obrazów w jednej iteracji treningowej.
epochs = 10000 

# Parametry CNN
CNN_EPOCHS_MRI = 11
BATCH_SIZE_MRI = 16
CNN_INPUT_SHAPE_MRI = (120,120,1)
VALIDATE_RATIO = 0.9   # ilosc w % ktora idzie na zbior testowy ze zbioru temp

CNN_MODELS_PATH_MRI = "MRI/CNN/MODEL"
CNN_PREDICT_PATH_MRI = "MRI/CNN/PREDICT/PREDICT_DATA"
PICKLE_DATA_ADHD_PATH = "MRI/PICKLE_DATA/ADHDImages.pkl"
PICKLE_DATA_ADHD_PATH_GAN = "PICKLE_DATA/adhdImages.pkl"
PICKLE_DATA_CONTROL_PATH_GAN = "PICKLE_DATA/controlImages.pkl"
GAN_MODELS_PATH = "MRI/GAN/MODEL"