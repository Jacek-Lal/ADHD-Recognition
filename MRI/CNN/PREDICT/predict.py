import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
import numpy as np

from EEG.PREDICT.eeg_read import checkResult
from MRI.config import *
from MRI.mri_plot import *
from MRI.mri_read import *

def print_index_ranges(y):
    adhd_indices = np.where(y == 1)[0]
    healthy_indices = np.where(y == 0)[0]

    if adhd_indices.size > 0:
        adhd_range = f"{adhd_indices[0]}-{adhd_indices[-1]}"
    else:
        adhd_range = "Brak indeksów"

    if healthy_indices.size > 0:
        healthy_range = f"{healthy_indices[0]}-{healthy_indices[-1]}"
    else:
        healthy_range = "Brak indeksów"

    print(f"Indeksy ADHD: {adhd_range}")
    print(f"Indeksy Zdrowe: {healthy_range}")

def predict_CNN(MODEL_NAME, cnn_model, cnn_predict):

    try:

        model = load_model(rf'{cnn_model}/{MODEL_NAME}.h5')

        X = readPickle(rf'{cnn_predict}/X_val_{MODEL_NAME}')

        y = readPickle(rf'{cnn_predict}/y_val_{MODEL_NAME}')

    except OSError as e:
        print(f'Błędna ścieżka do modelu {e}')
        return

    print_index_ranges(y)

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

    print(f"Wynik na całym zbiorze walidacyjnym: {accuracy*100:.4f} %")