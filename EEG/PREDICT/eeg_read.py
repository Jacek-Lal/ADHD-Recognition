import os
from scipy.io import loadmat
import math
import numpy as np

import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
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

def checkResult(predictions, threshold = 0.5):

    predictions[predictions>threshold] = 1
    predictions[predictions<=threshold] = 0

    mean = np.mean(predictions)

    if mean > threshold:
        print(f"Wynik pacjenta: ADHD, z prawdopodobieństwem: {np.round(mean*100,2)}%")
        return np.round(mean*100,2), "ADHD"
    else:
        print(f"Wynik pacjenta: ZDROWY, z prawdopodobieństwem: {np.abs(np.round((1-mean)*100,2))}%")
        return np.abs(np.round((1-mean)*100,2)), "ZDROWY"