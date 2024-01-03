import os
import nibabel as nib
import csv
from config import *
import pickle
import numpy as np

def read_nii_file(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    
    return data[:, :, 15, 60:70].astype(np.float16)    # slice nr 15, 10 próbek w czasie od 60 do 70

def getAdhdLabels(file_path):
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader, None)
        count = 0
        for row in reader:
            hasAdhd = int(row[5])
            count += hasAdhd
            data.append(hasAdhd)
            
    return data


def getTaskMRI(task, patientsNumber, patientType): # patientType: 0 for control, 1 for adhd
    data = []
    adhdLabels = getAdhdLabels(PATIENTS_DATA_PATH)

    for sub_num in range(1, patientsNumber + 1):

        hasAdhd = adhdLabels[sub_num - 1]

        if hasAdhd != patientType: continue
        
        sub_folder_name = f'sub-{sub_num:02d}'
        sub_folder_path = os.path.join(MRI_DATA_PATH, sub_folder_name, 'ses-T1', 'func')

        if not os.path.exists(sub_folder_path):
            print("Path does not exist")
            continue

        file = None
        for f in os.listdir(sub_folder_path):
            if f.endswith(f"{task}_bold.nii.gz"):
                file = f

        if file == None:
            print(f"There is no task {task} in this folder")
            continue
        
        nii_path = os.path.join(sub_folder_path, file)
        patient_images = read_nii_file(nii_path)
        for i in range(patient_images.shape[-1]):
            data.append(patient_images[:,:,i])

    return data


def load_images_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    images = []
    labels = []
    for patient in data:
        patientData = patient['data']
        for i in range(patientData.shape[-1]):
            images.append(patientData[:,:,:,i])
            labels.append(patient['hasAdhd'])

    return np.array(images), np.array(labels)