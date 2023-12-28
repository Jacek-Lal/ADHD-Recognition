import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from config import *
# from TRAIN.eeg_read import *
# from TRAIN.plots import *

import tensorflow as tf

# Sprawdź dostępność GPU
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print("TensorFlow korzysta z karty graficznej.")
    # Jeśli chcesz zobaczyć szczegółowe informacje o dostępnych GPU, odkomentuj poniższą linię.
    # print("Dostępne GPU:", gpu_devices)
else:
    print("TensorFlow korzysta z CPU.")