# Parametry CNN
CNN_EPOCHS_MRI = 11
CNN_BATCH_SIZE_MRI = 32
CNN_INPUT_SHAPE_MRI = (120,120,1)
CNN_LEARNING_RATE_MRI = 0.0002

GAN_EPOCHS_MRI = 2000
GAN_BATCH_SIZE_MRI = 32
GAN_INPUT_SHAPE_MRI = (120,120,1)
GAN_LEARNING_RATE_MRI = 0.0002

VALIDATE_RATIO = 0.9   # ilosc w % ktora idzie na zbior testowy ze zbioru temp


#SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
CNN_MODELS_PATH_MRI = "MRI/CNN/MODEL"
CNN_PREDICT_PATH_MRI = "MRI/CNN/PREDICT/PREDICT_DATA"
PICKLE_DATA_ADHD_PATH = "MRI/PICKLE_DATA/ADHDImages.pkl"
PICKLE_DATA_CONTROL_PATH = "MRI/PICKLE_DATA/controlImages.pkl"
GAN_MODELS_PATH = "MRI/GAN/MODEL"

def set_cnn_epochs_mri(new_value):
    global CNN_EPOCHS_MRI
    CNN_EPOCHS_MRI = new_value

def set_cnn_batch_size_mri(new_value):
    global CNN_BATCH_SIZE_MRI
    CNN_BATCH_SIZE_MRI = new_value

def set_cnn_learning_rate_mri(new_value):
    global CNN_LEARNING_RATE_MRI
    CNN_LEARNING_RATE_MRI = new_value

def set_cnn_input_shape_mri(new_value):
    global CNN_INPUT_SHAPE_MRI
    CNN_INPUT_SHAPE_MRI = new_value

def set_validate_ratio_mri(new_value):
    global VALIDATE_RATIO
    VALIDATE_RATIO = new_value

def set_gan_epochs_mri(new_value):
    global GAN_EPOCHS_MRI
    GAN_EPOCHS_MRI = new_value

def set_gan_batch_size_mri(new_value):
    global GAN_BATCH_SIZE_MRI
    GAN_BATCH_SIZE_MRI = new_value

def set_gan_learning_rate_mri(new_value):
    global GAN_LEARNING_RATE_MRI
    GAN_LEARNING_RATE_MRI = new_value

def set_gan_input_shape_mri(new_value):
    global GAN_INPUT_SHAPE_MRI
    GAN_INPUT_SHAPE_MRI = new_value