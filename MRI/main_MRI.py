import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from CNN.TRAIN.train import *
from CNN.PREDICT.predict import *
from GAN.TRAIN.train import *
from GAN.GENERATE.generate import *

gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print("TensorFlow korzysta z karty graficznej.")
    print("Dostępne GPU:", gpu_devices)
else:
    print("TensorFlow korzysta z CPU.")

# uruchamia trening
#train_CNN(save=False)

# uruchamia predict
# MODEL_CNN_NAME = "0.9478"
#
# predict_CNN(MODEL_CNN_NAME)

# uruchamia trening GAN
# train_GAN(save=False, data_type="ADHD")
