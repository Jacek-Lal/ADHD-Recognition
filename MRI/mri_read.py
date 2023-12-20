import os
import nibabel as nib
import numpy as np

MRI_DATA_PATH = './MRI'


def read_nii_file(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data


def read_MRI_file(root_folder):
    tasks = [
        'task-SLD',
        'task-SLI',
        'task-SSD',
        'task-SSI',
        'task-VLD',
        'task-VLI',
        'task-VSD',
        'task-VSI',
    ]

    data_dict = {}

    for task in tasks:
        task_data = {}
        for sub_num in range(1, 80):
            sub_folder = f'sub-{sub_num:02d}'
            sub_path = os.path.join(root_folder, sub_folder, 'ses-T1', 'func')
            if os.path.exists(sub_path):
                nii_files = [f for f in os.listdir(sub_path) if f.startswith(f'{sub_folder}_ses-T1_{task}_bold.nii.gz')]
                if nii_files:
                    task_data[sub_folder] = []
                    for nii_file in nii_files:
                        nii_path = os.path.join(sub_path, nii_file)
                        nii_data = read_nii_file(nii_path)
                        task_data[sub_folder].append(np.array(nii_data))

        data_dict[task] = task_data

    return data_dict


MRI_data = read_MRI_file(MRI_DATA_PATH)
print(MRI_data)

