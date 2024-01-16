from MRI.mri_read import *
from MRI.mri_filter import *
from MRI.CNN.TRAIN.train_model import *
from MRI.config import *
from MRI.GAN.GENERATE.generate import *

def train_CNN(save, pickle_data, adhd, control, cnn_predict):
    # SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
    ADHD = readPickle(rf'{pickle_data}/adhdImages.pkl')

    CONTROL = readPickle(rf'{pickle_data}/controlImages.pkl')

    ADHD_trimmed = trim(ADHD)

    CONTROL_trimmed = trim(CONTROL)

    ADHD_normalized = normalize(ADHD_trimmed)

    CONTROL_normalized = normalize(CONTROL_trimmed)

    #ADHD_GAN = generate_GAN("ADHD_GAN",im_amount=len(ADHD_normalized)*10)

    #CONTROL_GAN = generate_GAN("CONTROL_GAN",im_amount=len(CONTROL_normalized)*10)

    #savePickle("/home/user/Desktop/ADHD-Recognition/MRI/PICKLE_DATA/ADHD_GENERATED", ADHD_GAN)

    #savePickle("/home/user/Desktop/ADHD-Recognition/MRI/PICKLE_DATA/CONTROL_GENERATED", CONTROL_GAN)

    try:
        ADHD_GAN = readPickle(rf'{adhd}')

        CONTROL_GAN = readPickle(rf'{control}')
    except Exception as e:
        print(r"Bledna sciezka do plikow 'GENERATED'")
        return


    ADHD_CONCAT, CONTROL_CONCAT = concatWithGan(ADHD_GAN, CONTROL_GAN, ADHD_normalized, CONTROL_normalized)

    X_train, y_train, X_test, y_test, X_val, y_val = prepareForCnn(ADHD_CONCAT, CONTROL_CONCAT)

    accuracy = CnnFit(X_train, y_train, X_test, y_test, save)

    print(f"accuracy: {accuracy}")

    if save == True:
        # SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
        savePickle(rf'{cnn_predict}/X_val_{round(accuracy, 4)}', X_val)

        savePickle(rf'{cnn_predict}/y_val_{round(accuracy, 4)}', y_val)
