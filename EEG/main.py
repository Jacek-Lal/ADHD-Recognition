import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from config import *
from TRAIN.eeg_read import *
from TRAIN.plots import *

ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

plot(ADHD_DATA, 5, 1)