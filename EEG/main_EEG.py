import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from TRAIN.train import *
from PREDICT.predict import *

import os

current_dir = os.path.dirname(__file__)

MODEL_NAME = "0.4562"
MODEL_PATH = rf'{current_dir}/MODEL'
TRAIN_PATH = rf'{current_dir}/TRAIN/TRAIN_DATA'
PREDICT_PATH = rf'{current_dir}/PREDICT/PREDICT_DATA'


gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print("TensorFlow korzysta z karty graficznej.")
    print("Dostępne GPU:", gpu_devices)
else:
    print("TensorFlow korzysta z CPU.")

#uruchamia trening
train(True, TRAIN_PATH, PREDICT_PATH)

#uruchamia predict
MODEL_NAME = "0.4562"

predict(MODEL_NAME, MODEL_PATH, PREDICT_PATH)