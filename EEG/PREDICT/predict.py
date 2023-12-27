import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib
from keras.models import load_model

from EEG.PREDICT.eeg_read import *
from EEG.PREDICT.eeg_filter import *
from EEG.config import *
from EEG.PREDICT.plots import *


PATIENT_DIR = 'EEG/PREDICT/PREDICT_DATA/ADHD/v15p.mat'

MODEL_NAME = ""

DATA = readEEGRaw(PATIENT_DIR)

DATA_FILTERED = filterEEGData(DATA)

DATA_CLIPPED = clipEEGData(DATA_FILTERED)

DATA_NORMALIZED = normalizeEEGData(DATA_CLIPPED)

DATA_FRAMED = frameDATA(DATA_NORMALIZED)

print(DATA_FRAMED.shape)

# model = load_model(f'{EEG/MODEL}/{MODEL_NAME}")
#
# predictions = model.predict()
#
# print(predictions)