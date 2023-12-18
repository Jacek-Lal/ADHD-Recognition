import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from config import CNN_INPUT_SHAPE, EEG_SUBFOLDERS, EEG_POS_PHRASE, EEG_NEG_PHRASE, EEG_DATA_PATH, CNN_POS_LABEL, CNN_NEG_LABEL, EEG_SIGNAL_FRAME_SIZE, CNN_TEST_RATIO, CUTOFFS
from scipy import signal
    
def readEEGRaw(folder_path):
    """
        Matlab files from directory to numpy 2d arrays

        :param folder_path: EEG folder file with unziped subfolders
        :return: 2 lists (ADHD_DATA, CONTROL_DATA) of 2d numpy arrays, each 19 x individual_sample_count
    """

    # Lista podfolderów
    subfolders = EEG_SUBFOLDERS
    
    # Listy na dane
    ADHD_DATA = []
    CONTROL_DATA = []

    # Pętla po podfolderach
    for subfolder in subfolders:
        current_folder = os.path.join(folder_path, subfolder)

        # Lista plików .mat w bieżącym podfolderze
        mat_files = [f for f in os.listdir(current_folder) if f.endswith('.mat')]

        # Pętla po plikach .mat
        for mat_file in mat_files:
            file_path = os.path.join(current_folder, mat_file)

            # Wczytanie danych z pliku .mat
            loaded_data = loadmat(file_path, mat_dtype=True)

            # Uzyskanie nazwy pliku bez rozszerzenia
            file_name, _ = os.path.splitext(mat_file)

            # Zapisanie danych do odpowiedniego słownika w zależności od grupy
            if EEG_POS_PHRASE in subfolder:
                arr = loaded_data[file_name]
                ADHD_DATA.append(arr.T)
            elif EEG_NEG_PHRASE in subfolder:
                arr = loaded_data[file_name]
                CONTROL_DATA.append(arr.T)

    return ADHD_DATA, CONTROL_DATA

def filterEEGData(ADHD_DATA, CONTROL_DATA):
    ADHD_BANDWIDTHS, CONTROL_BANDWIDTHS = [], []
    fs = 128
    order = 4
    
    for cutoff in CUTOFFS:
        ADHD_FILTERED = []
        CONTROL_FILTERED = []
        
        low_cutoff = cutoff[0]
        high_cutoff = cutoff[1]
        b, a = signal.butter(order, [low_cutoff/(0.5*fs), high_cutoff/(0.5*fs)], btype='band')
        
        for i in range(len(ADHD_DATA)):
            ADHD_FILTERED.append(signal.filtfilt(b, a, ADHD_DATA[i])) 
            
        for i in range(len(CONTROL_DATA)):
            CONTROL_FILTERED.append(signal.filtfilt(b, a, CONTROL_DATA[i])) 
        
        ADHD_BANDWIDTHS.append(ADHD_FILTERED)
        CONTROL_BANDWIDTHS.append(CONTROL_FILTERED)
        
    return ADHD_BANDWIDTHS, CONTROL_BANDWIDTHS

def normalizeEEGData(ADHD_DATA, CONTROL_DATA):

    ADHD_NORMALIZED, CONTROL_NORMALIZED = [],[]
    num_patients_A = len(ADHD_DATA)
    num_patients_C = len(CONTROL_DATA)
    num_channels = CNN_INPUT_SHAPE[0]

    for i in range(num_patients_A):
        channels = []
        for j in range(num_channels):
            min_value = np.min(ADHD_DATA[i][j]).astype(np.float64)
            max_value = np.max(ADHD_DATA[i][j]).astype(np.float64)
            channels.append(((ADHD_DATA[i][j] - min_value) / (max_value - min_value)).astype(np.float64))
        ADHD_NORMALIZED.append(channels)

    for i in range(num_patients_C):
        channels = []
        for j in range(num_channels):
            min_value = np.min(CONTROL_DATA[i][j]).astype(np.float64)
            max_value = np.max(CONTROL_DATA[i][j]).astype(np.float64)
            channels.append(((CONTROL_DATA[i][j] - min_value) / (max_value - min_value)).astype(np.float64))
        CONTROL_NORMALIZED

    return ADHD_NORMALIZED, CONTROL_NORMALIZED

def framedEEGData(dataList, frameSize):
    """
        Turn patient EEG list of data (list of 2D matrixes) to 3D matrix with set sample size

        :param  dataList: List of 2D matrixes preferable set as [number_of_EEG_electrodes][electrode_samples]
        :param  frameSize: Size of one frame of samples
        :return: 3D matrix of size [num_of_total_data_frames][number_of_EEG_electrodes][frameSize * time]
    """
    result = []

    for matrix in dataList:
        num_rows, num_samples = matrix.shape
        num_frames = (num_samples // (frameSize))

        divided_matrix = np.array_split(matrix[:, :num_frames * frameSize], num_frames, axis=1)
        result.extend(divided_matrix)

    return np.array(result)

def getCNNData():
    ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

    ADHD_NORM, CONTROL_NORM = normalizeEEGData(ADHD_DATA, CONTROL_DATA)

    ADHD_FRAMED = framedEEGData(ADHD_NORM, EEG_SIGNAL_FRAME_SIZE)
    CONTROL_FRAMED = framedEEGData(CONTROL_NORM, EEG_SIGNAL_FRAME_SIZE)

    labelList = [CNN_POS_LABEL] * len(ADHD_FRAMED) + [CNN_NEG_LABEL] * len(CONTROL_FRAMED)

    X_DATA = np.concatenate((ADHD_FRAMED, CONTROL_FRAMED), axis=0)
    Y_DATA = np.array(labelList)

    X_train, X_test, y_train, y_test = train_test_split(X_DATA, Y_DATA, test_size=CNN_TEST_RATIO, random_state=42)

    return X_train, X_test, y_train, y_test