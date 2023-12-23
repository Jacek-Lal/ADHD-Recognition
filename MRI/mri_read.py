import os
import time
import nibabel as nib

MRI_DATA_PATH = r"MRI\files"


def read_nii_file(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def getMRIData(root_folder_name):
    tasks = [
        'SLD',
        'SLI',
        'SSD',
        'SSI',
        'VLD',
        'VLI',
        'VSD',
        'VSI',
    ]

    data_dict = {task: [] for task in tasks}
    
    for sub_num in range(1, 80):
        
        t1 = time.time()
        
        sub_folder_name = f'sub-{sub_num:02d}'
        sub_folder_path = os.path.join(root_folder_name, sub_folder_name, 'ses-T1', 'func')
        
        if not os.path.exists(sub_folder_path): 
            print("Path does not exist")
            continue

        nii_files = [f for f in os.listdir(sub_folder_path) if f.endswith(f"_bold.nii.gz")]
        
        if len(nii_files) == 0: 
            print("There are no .nii files in this folder")
            continue
        
        for file in nii_files:
            nii_path = os.path.join(sub_folder_path, file)
            nii_file_data = read_nii_file(nii_path)
            task = next(part.split('-')[1] for part in nii_path.split("_") if part.startswith('task-'))
            data_dict[task].append(nii_file_data)
        
        t2 = time.time()    
        print(f"Tasks from sub {sub_num} added in {round(t2-t1, 2)} s")
            
    return data_dict


MRI_data = getMRIData(MRI_DATA_PATH)
print(MRI_data)

