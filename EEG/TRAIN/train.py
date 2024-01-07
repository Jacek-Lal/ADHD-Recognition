import os
import sys

# Add the directory containing config.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from EEG.config import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from EEG.TRAIN.eeg_read import *
from EEG.TRAIN.train_model import *

def train():
    ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

    ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_DATA, CONTROL_DATA)

    ADHD_CLIPPED, CONTROL_CLIPPED = clipEEGData(ADHD_FILTERED, CONTROL_FILTERED)

    ADHD_NORMALIZED, CONTROL_NORMALIZED = normalizeEEGData(ADHD_CLIPPED, CONTROL_CLIPPED)

    X_train, y_train, X_test, y_test = prepareForCNN(ADHD_NORMALIZED, CONTROL_NORMALIZED)

    accuracy = CnnFit(X_train, y_train, X_test, y_test)

    print(f"accuracy: {accuracy}")
