import gc
from pympler import asizeof
import pickle
from mri_read import *
from config import *
from mri_plot import *

adhdLabels = getAdhdLabels(PATIENTS_DATA_PATH)
patientsNumber = 11

# data = getTaskMRI(TASKS[0], adhdLabels, patientsNumber) # List[{data: np.array, hasAdhd: int}]
# print(f"Shape: {data[0]['data'].shape}, {data[1]['data'].shape}, {data[2]['data'].shape}")
# print(f"Size of data: {asizeof.asizeof(data) / (1024.0**3):.2f} GB")
# print(f"Subs in task: {len(data)}")

with open('lista.pkl', 'rb') as f:
    data = pickle.load(f)

plot_mri(data[0]['data'])


del data
gc.collect()