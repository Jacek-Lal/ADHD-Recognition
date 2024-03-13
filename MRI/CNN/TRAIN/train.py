from MRI.mri_read import *
from MRI.mri_filter import *
from MRI.CNN.TRAIN.train_model import *
from MRI.config import *
from MRI.GAN.GENERATE.generate import *

def train_CNN(save):
    # SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
    ADHD = readPickle(f'../PICKLE_DATA/adhdImages.pkl')

    CONTROL = readPickle('../PICKLE_DATA/controlImages.pkl')

    ADHD_trimmed = trim(ADHD)

    CONTROL_trimmed = trim(CONTROL)

    ADHD_normalized = normalize(ADHD_trimmed)

    CONTROL_normalized = normalize(CONTROL_trimmed)

    #ADHD_GAN = generate_GAN("ADHD_GAN",im_amount=len(ADHD_normalized)*10)

    #CONTROL_GAN = generate_GAN("CONTROL_GAN",im_amount=len(CONTROL_normalized)*10)

    #savePickle("/home/user/Desktop/ADHD-Recognition/MRI/PICKLE_DATA/ADHD_GENERATED", ADHD_GAN)

    #savePickle("/home/user/Desktop/ADHD-Recognition/MRI/PICKLE_DATA/CONTROL_GENERATED", CONTROL_GAN)
    # SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
    ADHD_GAN = readPickle(f'../GENERATED_ADHD')

    CONTROL_GAN = readPickle(f'../GENERATED_CONTROL')


    ADHD_CONCAT, CONTROL_CONCAT = concatWithGan(ADHD_GAN, CONTROL_GAN, ADHD_normalized, CONTROL_normalized)

    X_train, y_train, X_test, y_test, X_val, y_val = prepareForCnn(ADHD_CONCAT, CONTROL_CONCAT)

    accuracy = CnnFit(X_train, y_train, X_test, y_test, save)

    print(f"accuracy: {accuracy}")

    if save == True:
        # SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
        savePickle(f'../PREDICT/PREDICT_DATA/X_val_{round(accuracy, 4)}', X_val)

        savePickle(f'../PREDICT/PREDICT_DATA/y_val_{round(accuracy, 4)}', y_val)