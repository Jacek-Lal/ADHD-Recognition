import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from CNN.TRAIN.train import *
from CNN.PREDICT.predict import *
from GAN.TRAIN.train import *

def MRI():

    gpu_devices = tf.config.list_physical_devices('GPU')

    if gpu_devices:
        print("TensorFlow korzysta z karty graficznej.")
        print("Dostępne GPU:", gpu_devices)
    else:
        print("TensorFlow korzysta z CPU.")


    # uruchamia trening GAN
    #train_GAN(save=True, data_type="CONTROL")

    # uruchamia trening
    #train_CNN(save=True)

    # uruchamia predict
    MODEL_CNN_NAME = "1.0"

    predict_CNN(MODEL_CNN_NAME)


