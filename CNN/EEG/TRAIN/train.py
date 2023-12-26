import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from eeg_read import *
from CNN.config import *
from CNN.EEG.TRAIN.train_model import *

ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_DATA, CONTROL_DATA)

ADHD_CLIPPED, CONTROL_CLIPPED = clipEEGData(ADHD_FILTERED, CONTROL_FILTERED)

ADHD_NORMALIZED, CONTROL_NORMALIZED = normalizeEEGData(ADHD_CLIPPED, CONTROL_CLIPPED)

X_train, y_train, X_test, y_test = prepareForCNN(ADHD_NORMALIZED, CONTROL_NORMALIZED)

model, accuracy = CnnFit1(X_train, y_train, X_test, y_test)

print(f"accuracy: {accuracy}")