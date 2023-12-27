import os
from scipy.io import loadmat
import math
import numpy as np

from EEG.config import *
    
def readEEGRaw(path):
    mat_data = loadmat(path, mat_dtype=True)

    file, _ = os.path.splitext(os.path.basename(path))

    return mat_data[file].T

def frameDATA(DATA):

    DATA_range = (math.floor(DATA.shape[1] / EEG_SIGNAL_FRAME_SIZE))

    DATA_framed = np.zeros((DATA_range, DATA.shape[0], EEG_SIGNAL_FRAME_SIZE))

    for i in range(DATA_range):
        DATA_framed[i, :, :] = DATA[:, i * EEG_SIGNAL_FRAME_SIZE: (i + 1) * EEG_SIGNAL_FRAME_SIZE]


    return np.reshape(DATA_framed,(DATA_framed.shape[0],DATA_framed.shape[1],DATA_framed.shape[2],1))

def checkResult(predictions):

    mean = np.mean(predictions, axis=0)

    if mean > 0.75:
        print(f"Wynik pacjenta: ADHD, z prawdopodobieństwem: {np.round(mean*100,2)}%")
    else:
        print(f"Wynik pacjenta: ZDROWY, z prawdopodobieństwem: {np.abs(np.round((1-mean)*100,2))}%")