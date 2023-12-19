import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from config import *
from scipy import signal
import copy
    
def readEEGRaw(folder_path):

    subfolders = EEG_SUBFOLDERS

    ADHD_DATA = []
    CONTROL_DATA = []

    for subfolder in subfolders:
        current_folder = os.path.join(folder_path, subfolder)


        mat_files = [f for f in os.listdir(current_folder) if f.endswith('.mat')]


        for mat_file in mat_files:
            file_path = os.path.join(current_folder, mat_file)


            loaded_data = loadmat(file_path, mat_dtype=True)

            file_name, _ = os.path.splitext(mat_file)

            if EEG_POS_PHRASE in subfolder:
                arr = loaded_data[file_name]
                ADHD_DATA.append(arr.T)
            elif EEG_NEG_PHRASE in subfolder:
                arr = loaded_data[file_name]
                CONTROL_DATA.append(arr.T)

    return ADHD_DATA, CONTROL_DATA

def filterEEGData(ADHD_DATA, CONTROL_DATA, band_type):
    order = 4
    cutoff = CUTOFFS[band_type]

    ADHD_FILTERED = []
    CONTROL_FILTERED = []
        
    low_cutoff = cutoff[0]
    high_cutoff = cutoff[1]
    b, a = signal.butter(order, [low_cutoff/(0.5*FS), high_cutoff/(0.5*FS)], btype='bandpass')
        
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

def framedEEGData(dataList, frameSize):

    result = []

    for matrix in dataList:
        num_rows, num_samples = matrix.shape
        num_frames = (num_samples // (frameSize))

        divided_matrix = np.array_split(matrix[:, :num_frames * frameSize], num_frames, axis=1)
        result.extend(divided_matrix)

    return np.array(result)

def getCNNData():
    ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

    ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_DATA, CONTROL_DATA,2)

    ADHD_NORM, CONTROL_NORM = normalizeEEGData(ADHD_FILTERED, CONTROL_FILTERED)

    ADHD_FRAMED = framedEEGData(ADHD_NORM, EEG_SIGNAL_FRAME_SIZE)
    CONTROL_FRAMED = framedEEGData(CONTROL_NORM, EEG_SIGNAL_FRAME_SIZE)

    labelList = [CNN_POS_LABEL] * len(ADHD_FRAMED) + [CNN_NEG_LABEL] * len(CONTROL_FRAMED)

    X_DATA = np.concatenate((ADHD_FRAMED, CONTROL_FRAMED), axis=0)
    Y_DATA = np.array(labelList)

    X_train, X_test, y_train, y_test = train_test_split(X_DATA, Y_DATA, test_size=CNN_TEST_RATIO, random_state=42)

    return X_train, X_test, y_train, y_test

