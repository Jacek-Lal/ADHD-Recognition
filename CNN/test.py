from eeg_read import *
from config import *
from plots import *
'''
import tensorflow as tf


devices = tf.config.list_physical_devices()

gpu_devices = [device for device in devices if 'GPU' in device.device_type]

if len(gpu_devices) > 0:
    print("TensorFlow używa karty graficznej do obliczeń.")
else:
    print("TensorFlow używa CPU lub nie ma dostępnej karty graficznej.")
    
'''

if __name__ == '__main__':
    ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)
    ADHD_NORMALIZED, CONTROL_NORMALIZED = normalizeEEGData(ADHD_DATA, CONTROL_DATA)


    for i in range(len(CUTOFFS)-1):

        ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_NORMALIZED, CONTROL_NORMALIZED, i)

        plot_frequency_band(ADHD_FILTERED[0][0], i)

    plt.show()
