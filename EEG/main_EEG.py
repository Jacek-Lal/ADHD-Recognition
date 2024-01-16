import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from TRAIN.train import *
from PREDICT.predict import *

import os

current_dir = os.path.dirname(__file__)

MODEL_NAME = "0.7974"
MODEL_PATH = rf'{current_dir}/MODEL'
TRAIN_PATH = rf'{current_dir}/TRAIN/TRAIN_DATA'
PREDICT_PATH = rf'{current_dir}/PREDICT/PREDICT_DATA'

def EEG():

    print("EEG")
    while True:
        try:
            choice = input('Wybierz opcję:   1-(uruchamia trening CNN)   2-(uruchamia predict CNN): ')
            if choice == '1':
                save = input('Wybierz opcję:   1-(zapisz model)   2-(nie zapisuj modelu): ')
                if save == '1':
                    train(True, TRAIN_PATH, PREDICT_PATH, MODEL_PATH)
                    break
                elif save == '2':
                    train(False, TRAIN_PATH, PREDICT_PATH, MODEL_PATH)
                    break
                else:
                    print("Niepoprawny wybór. Wprowadź 1 lub 2.")
            elif choice == '2':
                while(True):
                    predict(MODEL_NAME, MODEL_PATH, PREDICT_PATH)
            else:
                print("Niepoprawny wybór. Wprowadź 1 lub 2.")
        except Exception as e:
            print(f"Wystąpił błąd: {e}")

EEG()