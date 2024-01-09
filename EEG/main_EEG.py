import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from TRAIN.train import *
from PREDICT.predict import *

gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print("TensorFlow korzysta z karty graficznej.")
    print("Dostępne GPU:", gpu_devices)
else:
    print("TensorFlow korzysta z CPU.")

#uruchamia trening
#train(save=True)

#uruchamia predict
# PATIENT_DIR = 'ADHD/v274'
#
MODEL_NAME = "0.4562"
#
predict(MODEL_NAME)