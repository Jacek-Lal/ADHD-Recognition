import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from eeg_read import *
from eeg_filter import *
from config import *
from plots import *

import tensorflow as tf

devices = tf.config.list_physical_devices()

gpu_devices = [device for device in devices if 'GPU' in device.device_type]

if len(gpu_devices) > 0:
    print("TensorFlow używa karty graficznej do obliczeń.")
else:
    print("TensorFlow używa CPU lub nie ma dostępnej karty graficznej.")
