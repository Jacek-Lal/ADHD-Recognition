import gc
from pympler import asizeof
import pickle
from mri_read import *
from config import *
from mri_plot import *
import matplotlib.pyplot as plt

task = "VLI"
patientsNumber = 79
dataControl = getTaskMRI(task, patientsNumber, 0) 
dataAdhd = getTaskMRI(task, patientsNumber, 1) 

# with open("controlImages.pkl", 'wb') as f1:
#         pickle.dump(dataControl, f1)
        
# with open("adhdImages.pkl", 'wb') as f2:
#         pickle.dump(dataAdhd, f2)