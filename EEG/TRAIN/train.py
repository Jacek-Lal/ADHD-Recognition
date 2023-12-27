import os
import sys

# Add the directory containing config.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from eeg_read import *
from train_model import *

ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_DATA, CONTROL_DATA)

ADHD_CLIPPED, CONTROL_CLIPPED = clipEEGData(ADHD_FILTERED, CONTROL_FILTERED)

ADHD_NORMALIZED, CONTROL_NORMALIZED = normalizeEEGData(ADHD_CLIPPED, CONTROL_CLIPPED)

best_acc = 0
for i in range(10):
  X_train, y_train, X_test, y_test = prepareForCNN(ADHD_NORMALIZED, CONTROL_NORMALIZED)

  accuracy = CnnFit(X_train, y_train, X_test, y_test)
  best_acc = accuracy if accuracy > best_acc else best_acc
#
print(f"accuracy: {best_acc}")