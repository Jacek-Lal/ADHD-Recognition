import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from MRI.CNN.TRAIN.train import *
from CNN.PREDICT.predict import *
from GAN.TRAIN.train import *

import os

current_dir = os.path.dirname(__file__)

MODEL_CNN_NAME = "1.0"
PICKLE_PATH = rf'{current_dir}/PICKLE_DATA'
ADHD_PATH = r'/home/user/Desktop/ADHD_GENERATED'    #na potrzeby korzystania z serwera musi być sztywna
CONTROL_PATH = r'/home/user/Desktop/CONTROL_GENERATED'  #na potrzeby korzystania z serwera musi być sztywna
CNN_PREDICT_PATH = rf'{current_dir}/CNN/PREDICT/PREDICT_DATA'
CNN_MODEL_PATH = rf'{current_dir}/CNN/MODEL'
GAN_MODEL_PATH = rf'{current_dir}/GAN/MODEL'

def MRI():

    print("MRI")
    choice = input('Wybierz opcje:   1-(uruchamia trening CNN)   2-(uruchamia predict CNN)   3-(uruchamia trening GAN):')



    if choice == '1':
        save = input('Wybierz opcje:   1-(zapisz model)   2-(nie zapisuj modelu):')
        if save == '1':
            train_CNN(True, PICKLE_PATH, ADHD_PATH, CONTROL_PATH, CNN_PREDICT_PATH)
        elif save == '2':
            train_CNN(False, PICKLE_PATH, ADHD_PATH, CONTROL_PATH, CNN_PREDICT_PATH)
    elif choice == '2':
        predict_CNN(MODEL_CNN_NAME, CNN_MODEL_PATH, CNN_PREDICT_PATH)
    elif choice == '3':

        type = input('Wybierz opcje:   1-(CONTROL)   2-(ADHD):')
        save = input('Wybierz opcje:   1-(zapisz model)   2-(nie zapisuj modelu):')
        if save == '1':
            if type == 'CONTROL':
                train_GAN(True, data_type="CONTROL", pickle=PICKLE_PATH, model=GAN_MODEL_PATH)
            elif type == 'ADHD':
                train_GAN(True, data_type="ADHD", pickle=PICKLE_PATH, model=GAN_MODEL_PATH)
        elif save == '2':
            if type == 'CONTROL':
                train_GAN(False, data_type="CONTROL", pickle=PICKLE_PATH, model=GAN_MODEL_PATH)
            elif type == 'ADHD':
                train_GAN(False, data_type="ADHD", pickle=PICKLE_PATH, model=GAN_MODEL_PATH)



#MRI()