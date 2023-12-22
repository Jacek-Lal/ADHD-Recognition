import numpy as np
from config import *
from scipy import signal
import copy
from scipy.signal import butter, lfilter

def highpassFilterEEG(ADHD_DATA, CONTROL_DATA):
    ADHD_FILTERED = []
    CONTROL_FILTERED = []

    b, a = butter(4, 1.0 / (0.5 * FS), btype='high', analog=False)
    for i in range(len(ADHD_DATA)):
        ADHD_FILTERED.append(lfilter(b, a, ADHD_DATA[i]))
    for i in range(len(CONTROL_DATA)):
        CONTROL_FILTERED.append(lfilter(b, a, CONTROL_DATA[i]))

    return ADHD_FILTERED, CONTROL_FILTERED


def filterEEGData(ADHD_DATA, CONTROL_DATA, band_type=2):
    order = 4
    cutoff = CUTOFFS[band_type]

    ADHD_FILTERED = []
    CONTROL_FILTERED = []

    low_cutoff = cutoff[0]
    high_cutoff = cutoff[1]
    b, a = signal.butter(order, [low_cutoff / (0.5 * FS), high_cutoff / (0.5 * FS)], btype='bandpass')

    for i in range(len(ADHD_DATA)):
        ADHD_FILTERED.append(signal.filtfilt(b, a, ADHD_DATA[i]))

    for i in range(len(CONTROL_DATA)):
        CONTROL_FILTERED.append(signal.filtfilt(b, a, CONTROL_DATA[i]))

    return ADHD_FILTERED, CONTROL_FILTERED

def normalizeEEGData(ADHD_DATA, CONTROL_DATA):
    num_patients_A = len(ADHD_DATA)
    num_patients_C = len(CONTROL_DATA)

    if (num_patients_A <= 1) or (num_patients_C <= 1):
        print("Ta funkcja oblicza dane dla więcej niż 1 pacjenta")

    ADHD_DATA_normalized = copy.deepcopy(ADHD_DATA)
    CONTROL_DATA_normalized = copy.deepcopy(CONTROL_DATA)

    for i in range(num_patients_A):
        for j in range(CNN_INPUT_SHAPE[0]):
            min_value = np.min(ADHD_DATA_normalized[i][j]).astype(np.float64)
            max_value = np.max(ADHD_DATA_normalized[i][j]).astype(np.float64)
            ADHD_DATA_normalized[i][j] = ((ADHD_DATA_normalized[i][j] - min_value) / (max_value - min_value)).astype(np.float64)

    for i in range(num_patients_C):
        for j in range(CNN_INPUT_SHAPE[0]):
            min_value = np.min(CONTROL_DATA_normalized[i][j]).astype(np.float64)
            max_value = np.max(CONTROL_DATA_normalized[i][j]).astype(np.float64)
            CONTROL_DATA_normalized[i][j] = ((CONTROL_DATA_normalized[i][j] - min_value) / (max_value - min_value)).astype(np.float64)

    return ADHD_DATA_normalized, CONTROL_DATA_normalized

def deleteMedianEEG(ADHD_DATA,CONTROL_DATA, median_level = 4):
    ADHD_MEDIANED = copy.deepcopy(ADHD_DATA)
    CONTROL_MEDIANED = copy.deepcopy(CONTROL_DATA)


    for i in range(len(ADHD_DATA)):
        for j in range(CNN_INPUT_SHAPE[0]):
            median = np.median(ADHD_DATA[i][j])
            median_mask = np.abs(median_level*median)
            channel_data = ADHD_DATA[i][j]
            ADHD_MEDIANED[i][j][(channel_data>median_mask) | (channel_data<-median_mask)] = median

    for i in range(len(CONTROL_DATA)):
        for j in range(CNN_INPUT_SHAPE[0]):
            median = np.median(CONTROL_DATA[i][j])
            median_mask = np.abs(median_level*median)
            channel_data = CONTROL_DATA[i][j]
            CONTROL_MEDIANED[i][j][(channel_data>median_mask) | (channel_data<-median_mask)] = median


    return ADHD_MEDIANED, CONTROL_MEDIANED


def clipEEGData(ADHD_DATA, CONTROL_DATA):
    num_patients_A = len(ADHD_DATA)
    num_patients_C = len(CONTROL_DATA)
    percentile = 99.8

    ADHD_CLIPPED = copy.deepcopy(ADHD_DATA)
    CONTROL_CLIPPED = copy.deepcopy(CONTROL_DATA)
    ADHD_TRESHOLDS = []
    CONTROL_TRESHOLDS = []

    for i in range(num_patients_A):
        for j in range(CNN_INPUT_SHAPE[0]):
            channel_data = ADHD_DATA[i][j]
            treshold = np.abs(np.percentile(channel_data, percentile))
            clipped_data = np.clip(channel_data, a_min=-treshold, a_max=treshold)
            ADHD_CLIPPED[i][j] = clipped_data
            ADHD_TRESHOLDS.append(treshold)

    for i in range(num_patients_C):
        for j in range(CNN_INPUT_SHAPE[0]):
            channel_data = CONTROL_DATA[i][j]
            treshold = np.abs(np.percentile(channel_data, percentile))
            clipped_data = np.clip(channel_data, a_min=-treshold, a_max=treshold)
            CONTROL_CLIPPED[i][j] = clipped_data
            CONTROL_TRESHOLDS.append(treshold)

    return ADHD_CLIPPED, CONTROL_CLIPPED