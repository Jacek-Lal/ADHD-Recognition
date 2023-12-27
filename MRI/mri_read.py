import os
import nibabel as nib
import csv
from config import *

def read_nii_file(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()

    return img

def getAdhdLabels(file_path):
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader, None)

        for row in reader:
            hasAdhd = int(row[5])
            data.append(hasAdhd)
    
    return data

def getTaskMRI(task, adhd_labels, patientsNumber):
    
    data = []

    for sub_num in range(1, patientsNumber + 1):
        
        hasAdhd = adhd_labels[sub_num - 1]
        
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
        nii_file_data = read_nii_file(nii_path)

        patient_data = {"data": nii_file_data, "hasAdhd": hasAdhd}
        
        data.append(patient_data)
            
    return data

