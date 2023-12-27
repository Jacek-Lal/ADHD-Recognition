import os
from scipy.io import loadmat
    
def readEEGRaw(path):
    mat_data = loadmat(path, mat_dtype=True)

    file, _ = os.path.splitext(os.path.basename(path))

    return mat_data[file].T