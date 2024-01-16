import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model

from EEG.PREDICT.eeg_read import *
from EEG.PREDICT.eeg_filter import *
from EEG.config import *
from MRI.mri_read import readPickle

def predict(MODEL_NAME, model_path, pickle_path):

    try:

        model = load_model(rf'{model_path}/{MODEL_NAME}.h5')

        X = readPickle(rf'{pickle_path}/X_val_{MODEL_NAME}')

        y = readPickle(rf'{pickle_path}/y_val_{MODEL_NAME}')

    except OSError as e:
        print(f'Błędna ścieżka do modelu {e}')
        return

    print(f"Indeksy ADHD{np.where(y==1)[0]}")

    print(f"Indeksy Zdrowe{np.where(y == 0)[0]}")

    while True:
        try:
            patient_number = int(input("Wybierz numer pacjenta: "))
            if patient_number < len(X) and patient_number >= 0:
                break
            else:
                print("Wpisz numer pacjenta w zakresie")
        except ValueError:
            print("Wpisz numer pacjenta zakresie")

    if y[patient_number] == 1:
        print("Wybrales ADHD")
    elif y[patient_number] == 0:
        print("Wybrales Zdrowy")



    DATA = X[patient_number]

    DATA_FILTERED = filterEEGData(DATA)

    DATA_CLIPPED = clipEEGData(DATA_FILTERED)

    DATA_NORMALIZED = normalizeEEGData(DATA_CLIPPED)

    DATA_FRAMED = frameDATA(DATA_NORMALIZED)

    predictions = model.predict(DATA_FRAMED)

    checkResult(predictions)
