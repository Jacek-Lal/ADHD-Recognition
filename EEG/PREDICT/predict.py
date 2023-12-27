import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from eeg_read import *
from eeg_filter import *
from keras.models import load_model
from EEG.config import *
from plots import *
import joblib

PATIENT_DIR = 'TRAIN_DATA/TRAIN_DATA/PREDICT/ADHD/v37p.mat'
MODEL_PATH = 'TRAIN_DATA/MODEL/model'

DATA = readEEGRaw(PATIENT_DIR)

DATA_FILTERED = filterEEGData(DATA)

DATA_CLIPPED = clipEEGData(DATA_FILTERED)

DATA_NORMALIZED = normalizeEEGData(DATA_CLIPPED)

# model = load_model(MODEL_PATH)
#
# predictions = model.predict()
#
# print(predictions)