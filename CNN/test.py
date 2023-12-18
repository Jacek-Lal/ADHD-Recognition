from training import check_saved_trained_models, save_trained_models, get_prepared_model
from eeg_read import *
from config import *
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from plots import *

ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)
ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_DATA, CONTROL_DATA)

bandwith = 0
patient = 10
channel = 10

plot(ADHD_DATA, patient, channel)
plot(ADHD_FILTERED[bandwith], patient, channel)
plt.show()

