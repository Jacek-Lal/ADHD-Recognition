import gc
from pympler import asizeof

from mri_read import *
from config import *

adhdLabels = getAdhdLabels(PATIENTS_DATA_PATH)

for task in TASKS:
    data = getTaskMRI(task, adhdLabels)
    
    print(f"Size of data: {asizeof.asizeof(data) / (1024.0**3):.2f} GB")
    print(f"Subs in task: {len(data)}")
    
    del data
    gc.collect()

