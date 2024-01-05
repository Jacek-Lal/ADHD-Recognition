import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
import numpy as np

from EEG.PREDICT.eeg_read import checkResult
from MRI.config import *
from MRI.mri_plot import *
from MRI.mri_read import *

def predict(MODEL_NAME):

    try:
        model = load_model(f'{CNN_MODELS_PATH_MRI}/{MODEL_NAME}.h5')

        X = readPickle(f"{CNN_PREDICT_PATH_MRI}/X_val_{MODEL_NAME}")

        y = readPickle(f"{CNN_PREDICT_PATH_MRI}/y_val_{MODEL_NAME}")

    except OSError as e:
        print("Błędna ścieżka do modelu")
        return

    print(f"Indeksy ADHD{np.where(y==1)[0]}")

    print(f"Indeksy Zdrowe{np.where(y == 0)[0]}")

    while True:
        try:
            image_number = int(input("Wybierz zdjecie: "))
            if image_number < X.shape[0] and image_number >= 0:
                break
            else:
                print("Wpisz numer zdjecia w zakresie")
        except ValueError:
            print("Wpisz numer zdjecia zakresie")

    if y[image_number] == 1:
        print("Wybrales ADHD")
    elif y[image_number] == 0:
        print("Wybrales Zdrowy")

    plot_mri(X[image_number])

    _ , accuracy = model.evaluate(X,y, verbose = 0)

    img_for_predict = X[image_number].reshape(1,X[image_number].shape[0],X[image_number].shape[1],1)

    predictions = model.predict(img_for_predict)

    checkResult(predictions)

    print(f"Wynik na całym zbiorze walidacyjnym: {accuracy:.4f}")