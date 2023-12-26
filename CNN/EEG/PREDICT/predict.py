import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from scipy.io import loadmat

PATIENT_DIR = 'CNN/EEG/PREDICT/ADHD/v37p.mat'
def loadonepatient(path):
    mat_data = loadmat(path, mat_dtype=True)
    file, _ = os.path.splitext(os.path.basename(path))
    mat_data = mat_data[file]
    mat_data = mat_data.T
    return mat_data

data = loadonepatient(PATIENT_DIR)

