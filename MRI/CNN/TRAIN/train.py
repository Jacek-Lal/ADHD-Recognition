from MRI.mri_read import *
from MRI.mri_filter import *
from MRI.CNN.TRAIN.train_model import *
from MRI.config import *
from MRI.GAN.GENERATE.generate import *

def train_CNN(save):

    ADHD = readPickle(PICKLE_DATA_ADHD_PATH)

    CONTROL = readPickle(PICKLE_DATA_CONTROL_PATH)

    ADHD_trimmed = trim(ADHD)

    CONTROL_trimmed = trim(CONTROL)

    ADHD_normalized = normalize(ADHD_trimmed)

    CONTROL_normalized = normalize(CONTROL_trimmed)

    ADHD_GAN = generate_GAN("ADHD_",im_amount=len(ADHD_normalized)*10)

    CONTROL_GAN = generate_GAN("CONTROL_0.6647",im_amount=len(CONTROL_normalized)*10)

    ADHD_CONCAT, CONTROL_CONCAT = concatWithGan(ADHD_GAN, CONTROL_GAN, ADHD_normalized, CONTROL_normalized)

    X_train, y_train, X_test, y_test, X_val, y_val = prepareForCnn(ADHD_CONCAT, CONTROL_CONCAT)

    #accuracy = CnnFit(X_train, y_train, X_test, y_test, save)

    #print(f"accuracy: {accuracy}")

    # if save == True:
    #
    #     savePickle(f"{CNN_PREDICT_PATH_MRI}/X_val_{round(accuracy, 4)}", X_val)
    #
    #     savePickle(f"{CNN_PREDICT_PATH_MRI}/y_val_{round(accuracy, 4)}", y_val)