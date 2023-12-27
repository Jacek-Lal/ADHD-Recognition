import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from EEG.TRAIN.eeg_read import *
from EEG.config import *
from EEG.TRAIN.train_model import *

ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_DATA, CONTROL_DATA)

ADHD_CLIPPED, CONTROL_CLIPPED = clipEEGData(ADHD_FILTERED, CONTROL_FILTERED)

ADHD_NORMALIZED, CONTROL_NORMALIZED = normalizeEEGData(ADHD_CLIPPED, CONTROL_CLIPPED)

X_train, y_train, X_test, y_test = prepareForCNN(ADHD_NORMALIZED, CONTROL_NORMALIZED)

# model, accuracy = CnnFit_test(X_train, y_train, X_test, y_test)
#
# print(f"accuracy: {accuracy}")