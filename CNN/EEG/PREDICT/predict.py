import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from eeg_read import *
from eeg_filter import *
from keras.models import load_model
from CNN.config import *

PATIENT_DIR = 'CNN/EEG/PREDICT/ADHD/v37p.mat'
MODEL_PATH = 'CNN/MODEL/model'


DATA = readEEGRaw(PATIENT_DIR)



DATA_FILTERED = filterEEGData(DATA)


DATA_CLIPPED = clipEEGData(DATA_FILTERED)


DATA_NORMALIZED = normalizeEEGData(DATA_CLIPPED)

# model = load_model(MODEL_PATH)
#
# predictions = model.predict()
#
# MED_FOR_0 = np.mean(predictions[:, 0])
#
# MED_FOR_1 = np.mean(predictions[:, 1])
#
# if MED_FOR_0 > MED_FOR_1:
#     print("Wykryto ADHD")
# elif MED_FOR_0 < MED_FOR_1:
#     print("Pacjent zdrowy")
# else:
#     print("Chuj wie, idz do lekarza")
