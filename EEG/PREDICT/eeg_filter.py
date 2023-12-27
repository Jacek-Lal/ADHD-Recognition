import numpy as np
from scipy import signal
import copy

import sys
import os
# Add the directory containing config.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import *

def filterEEGData(DATA, band_type=2):

    order = 4
    cutoff = CUTOFFS[band_type]

    DATA_filtered = copy.deepcopy(DATA)

    low_cutoff = cutoff[0]
    high_cutoff = cutoff[1]

    b, a = signal.butter(order, [low_cutoff / (0.5 * FS), high_cutoff / (0.5 * FS)], btype='bandpass')

    return signal.filtfilt(b, a, DATA_filtered)

def normalizeEEGData(DATA):

    DATA_normalized = copy.deepcopy(DATA)

    for i in range(CNN_INPUT_SHAPE[0]):
        min_value = np.min(DATA_normalized[i]).astype(np.float64)
        max_value = np.max(DATA_normalized[i]).astype(np.float64)
        DATA_normalized[i] = ((DATA_normalized[i] - min_value) / (max_value - min_value)).astype(np.float64)

    return DATA_normalized

def clipEEGData(DATA):

    percentile = 99.8

    DATA_CLIPPED = copy.deepcopy(DATA)

    for i in range(CNN_INPUT_SHAPE[0]):
        channel_data = DATA[i]
        treshold = np.abs(np.percentile(channel_data, percentile))
        clipped_data = np.clip(channel_data, a_min=-treshold, a_max=treshold)
        DATA_CLIPPED[i] = clipped_data

    return DATA_CLIPPED