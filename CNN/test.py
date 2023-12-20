import numpy as np

from eeg_read import *
from config import *
from plots import *
import matplotlib.pyplot as plt
import copy



ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_DATA, CONTROL_DATA)

plot(ADHD_FILTERED, 5, 6)
plt.show()