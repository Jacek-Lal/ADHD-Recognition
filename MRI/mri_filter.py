import numpy as np
import copy

def normalize(data):
    normalized = copy.deepcopy(data)

    for i in range(len(data)):
        min = np.min(data[i])
        max = np.max(data[i])
        normalized[i] = (data[i]-min)/(max-min)

    return normalized

def checkdim(data):
    for i in range(len(data)):
        rows, columns = data[i].shape
        if rows != columns:
            print(f"data: {i} ma wymiary nie bedace kwadratem: {rows, columns}")

def trim(data, nr_rows = 4):
    trimmed = copy.deepcopy(data)
    for i in range(len(data)):
        trimmed[i] = data[i][nr_rows:-nr_rows]

    return trimmed