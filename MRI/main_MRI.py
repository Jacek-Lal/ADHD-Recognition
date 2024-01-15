import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from MRI.CNN.TRAIN.train import *
from CNN.PREDICT.predict import *
from GAN.TRAIN.train import *

def MRI():

    gpu_devices = tf.config.list_physical_devices('GPU')

    if gpu_devices:
        print("TensorFlow korzysta z karty graficznej.")
        print("Dostępne GPU:", gpu_devices)
    else:
        print("TensorFlow korzysta z CPU.")

    print(os.getcwd())
    print("MRI")
    choice = input('Wybierz opcje:   1-(uruchamia trening CNN)   2-(uruchamia predict CNN)   3-(uruchamia trening GEN):')

    if choice == '1':
        train_CNN(save=True)
    elif choice == '2':
        MODEL_CNN_NAME = "1.0"
        predict_CNN(MODEL_CNN_NAME)
    elif choice == '3':
        train_GAN(save=True, data_type="CONTROL")





