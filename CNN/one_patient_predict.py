from eeg_read import *
from eeg_filter import *
from config import *
from plots import *
import os
import numpy as np

def loadPatientData(path):
    mat_data = loadmat(path)

    plik = os.path.basename(path)
    nazwa_pliku = os.path.splitext(plik)[0]

    if nazwa_pliku in mat_data:
         arr = mat_data[nazwa_pliku]
         return arr.T
    else:
         return None
    
def filterEEGData(data, band_type=2):
    order = 4
    cutoff = CUTOFFS[band_type]

    data_filtered = []

    low_cutoff = cutoff[0]
    high_cutoff = cutoff[1]
    b, a = signal.butter(order, [low_cutoff / (0.5 * FS), high_cutoff / (0.5 * FS)], btype='bandpass')

    for i in range(len(data)):
        data_filtered.append(signal.filtfilt(b, a, data[i]))

    return data_filtered

def normalizeEEGData(data):
    num_patients = len(data)

    if (num_patients <= 1):
        print("Ta funkcja oblicza dane dla więcej niż 1 pacjenta")

    data_normalized = copy.deepcopy(data)

    for j in range(CNN_INPUT_SHAPE[0]):
        min_value = np.min(data_normalized[j]).astype(np.float64)
        max_value = np.max(data_normalized[j]).astype(np.float64)
        data_normalized[j] = ((data_normalized[j] - min_value) / (max_value - min_value)).astype(np.float64)

    return data_normalized

def clipEEGData(ADHD_DATA):
    percentile = 99.8

    ADHD_CLIPPED = copy.deepcopy(ADHD_DATA)
    ADHD_TRESHOLDS = []

    for j in range(CNN_INPUT_SHAPE[0]):
        channel_data = ADHD_DATA[j]
        treshold = np.abs(np.percentile(channel_data, percentile))
        clipped_data = np.clip(channel_data, a_min=-treshold, a_max=treshold)
        ADHD_CLIPPED[j] = clipped_data
        ADHD_TRESHOLDS.append(treshold)

    return ADHD_CLIPPED

def framedEEGData(data, frame_size):
    data = np.array(data)
    num_samples = len(data[0])
    num_frames = (num_samples // (frame_size))

    reshaped_data = data[:, :num_frames * frame_size].reshape(num_frames, 19, frame_size)

    return reshaped_data

def makePrediction(model, data):
    output = model.predict(data)

    chance = np.exp(np.mean(np.log(output)))
    result = True if chance >= 0.5 else False

    return result, chance

def predictPatient(model, path):
    data = loadPatientData(path)

    data_filtered = filterEEGData(data)

    data_clipped = clipEEGData(data_filtered)

    data_norm = normalizeEEGData(data_clipped)

    data_framed = framedEEGData(data_norm, EEG_SIGNAL_FRAME_SIZE)

    result, chance = makePrediction(model, data_framed)

    return result, chance