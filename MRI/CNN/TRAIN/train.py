from MRI.mri_read import *
from MRI.mri_filter import *
from MRI.CNN.TRAIN.train_model import *
from MRI.config import *

def train(save):

    ADHD = readPickle(PICKLE_DATA_ADHD_PATH)

    CONTROL = readPickle(PICKLE_DATA_CONTROL_PATH)

    ADHD_trimmed = trim(ADHD)

    CONTROL_trimmed = trim(CONTROL)

    ADHD_normalized = normalize(ADHD_trimmed)

    CONTROL_normalized = normalize(CONTROL_trimmed)

    X_train, y_train, X_test, y_test, X_val, y_val = prepareForCnn(ADHD_normalized, CONTROL_normalized)

    accuracy = CnnFit(X_train, y_train, X_test, y_test, save)

    print(f"accuracy: {accuracy}")

    if save == True:

        savePickle(f"{CNN_PREDICT_PATH_MRI}/X_val_{round(accuracy, 4)}", X_val)

        savePickle(f"{CNN_PREDICT_PATH_MRI}/y_val_{round(accuracy, 4)}", y_val)