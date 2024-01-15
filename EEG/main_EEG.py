import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from TRAIN.train import *
from PREDICT.predict import *


def EEG():
    gpu_devices = tf.config.list_physical_devices('GPU')

    if gpu_devices:
        print("TensorFlow korzysta z karty graficznej.")
        print("Dostępne GPU:", gpu_devices)
    else:
        print("TensorFlow korzysta z CPU.")

    print("EEG")
    choice = input('Wybierz opcje:   1-(uruchamia trening CNN)   2-(uruchamia predict CNN):')

    if choice == '1':
        train(True, rf'EEG/TRAIN/TRAIN_DATA', rf'EEG/PREDICT/PREDICT_DATA')
    elif choice == '2':
        MODEL_NAME = "0.4562"
        predict(MODEL_NAME, rf'EEG/MODEL', rf'EEG/PREDICT/PREDICT_DATA')
