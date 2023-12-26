import os
from scipy.io import loadmat
    
def readEEGRaw(path):
    mat_data = loadmat(path, mat_dtype=True)
    file, _ = os.path.splitext(os.path.basename(path))
    mat_data = mat_data[file]
    mat_data = mat_data.T

    return mat_data