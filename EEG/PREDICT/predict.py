import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model

from eeg_read import *
from eeg_filter import *
import sys

# Add the directory containing config.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import *
from plots import *


PATIENT_DIR = 'CONTROL/v307'

MODEL_NAME = "0.8828"

DATA = readEEGRaw(f'EEG/PREDICT/PREDICT_DATA/{PATIENT_DIR}.mat')

DATA_FILTERED = filterEEGData(DATA)

DATA_CLIPPED = clipEEGData(DATA_FILTERED)

DATA_NORMALIZED = normalizeEEGData(DATA_CLIPPED)

DATA_FRAMED = frameDATA(DATA_NORMALIZED)

model = load_model(f'{CNN_MODELS_PATH}/{MODEL_NAME}.h5')

predictions = model.predict(DATA_FRAMED)

checkResult(predictions)