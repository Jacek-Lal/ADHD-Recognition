import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from config import *
from eeg_filter import *
    
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

    X_train, X_test, y_train, y_test = train_test_split(X_DATA, Y_DATA, test_size=CNN_TEST_RATIO)

    return X_train, X_test, y_train, y_test