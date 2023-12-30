import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print("TensorFlow korzysta z karty graficznej.")
    print("Dostępne GPU:", gpu_devices)
else:
    print("TensorFlow korzysta z CPU.")
