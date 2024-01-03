import numpy as np

def normalize(data):
    min = np.min(data)
    max = np.max(data)
    return (data-min)/(max-min)