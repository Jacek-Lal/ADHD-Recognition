import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model

from EEG.PREDICT.eeg_read import *
from EEG.PREDICT.eeg_filter import *
from EEG.config import *
from EEG.PREDICT.plots import *


PATIENT_DIR = 'CONTROL/v129'

MODEL_NAME = "0.8828"

DATA = readEEGRaw(f'EEG/PREDICT/PREDICT_DATA/{PATIENT_DIR}.mat')

DATA_FILTERED = filterEEGData(DATA)

DATA_CLIPPED = clipEEGData(DATA_FILTERED)

DATA_NORMALIZED = normalizeEEGData(DATA_CLIPPED)

DATA_FRAMED = frameDATA(DATA_NORMALIZED)

model = load_model(f'{CNN_MODELS_PATH}/{MODEL_NAME}.h5')

predictions = model.predict(DATA_FRAMED)

checkResult(predictions)