import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from eeg_read import *
from eeg_filter import *
from config import *
from plots import *

adhd_data, _ = readEEGRaw(EEG_DATA_PATH)

adhd_data = adhd_data[0]

print((adhd_data.reshape(-1,adhd_data.shape[0], adhd_data.shape[1])).shape)