import numpy as np

from eeg_read import *
from config import *
from plots import *
import matplotlib.pyplot as plt
import copy

def deleteMedianEEG(ADHD_DATA,CONTROL_DATA, median_level = 600):
    ADHD_MEDIANED = copy.deepcopy(ADHD_DATA)
    CONTROL_MEDIANED = copy.deepcopy(CONTROL_DATA)


    for i in range(len(ADHD_DATA)):
        for j in range(CNN_INPUT_SHAPE[0]):
            median = np.median(ADHD_DATA[i][j])
            median_mask = median_level*median
            channel_data = ADHD_DATA[i][j]
            ADHD_MEDIANED[i][j][(channel_data>median_mask) | (channel_data<-median_mask)] = median

    for i in range(len(CONTROL_DATA)):
        for j in range(CNN_INPUT_SHAPE[0]):
            median = np.median(CONTROL_DATA[i][j])
            median_mask = median_level*median
            channel_data = CONTROL_DATA[i][j]
            CONTROL_MEDIANED[i][j][(channel_data<median_mask) | (channel_data>-median_mask)] = median

    print(f"Próg odcięcia: {median_mask}")

    plot(ADHD_DATA, 11, 0)
    plt.show()

    return ADHD_MEDIANED, CONTROL_MEDIANED

ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_DATA, CONTROL_DATA)

plot(ADHD_FILTERED, 5, 6)
plt.show()