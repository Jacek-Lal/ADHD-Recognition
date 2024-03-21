from PyQt5 import uic
import os

from PyQt5.QtWidgets import QFileDialog, QVBoxLayout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
#from pyedflib import EdfReader
#from nibabel import load
from scipy.io import loadmat
from pandas import read_csv
import numpy as np
#from keras.models import load_model

#from EEG.PREDICT.eeg_filter import filterEEGData, clipEEGData, normalizeEEGData
#from EEG.PREDICT.eeg_read import frameDATA, checkResult

current_dir = os.path.dirname(__file__)
UI_PATH = rf'{current_dir}/UI'
parent_directory = os.path.dirname(current_dir)
FILE_TYPES = ["mat","csv",'edf','nii.gz','nii']



class DoctorViewController:
    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.ui = uic.loadUi(rf'{parent_directory}/UI/doctorView.ui', mainWindow)
        self.addEvents()
        self.filePaths = None

    def addEvents(self):
        self.ui.loadDataBtn.clicked.connect(self.getFilePaths)

    def getFilePaths(self):
        options = QFileDialog.Options()
        fileFilter = ";;".join([f"{ext} files (*.{ext})" for ext in FILE_TYPES])
        self.filePaths, _ = QFileDialog.getOpenFileNames(self.mainWindow, "Choose files", "./EEG/TRAIN/TRAIN_DATA/ADHD", "", options=options)
        self.processFiles()

    def processFiles(self):
        data = np.array([])

        for path in self.filePaths:

            # załaduj plik ( zależnie od typu inaczej, po to tyle if'ów)
            # zdecyduj czy eeg czy mri  (na podstawie struktury/rozszerzenia)
            # wybierz z niego potrzebne dane
            # wybierz model na podstawie struktury danych (np dla EEG różna ilość kanałów, dla mri różna płaszczyzna mózgu)
            # wrzuć dane w model
            # zwróc wynik

            if path.endswith('.edf'):
                print("EDF")
                #file = EdfReader(path)
                #signalsNum = file.signals_in_file
                #print(type(file.readsignal(1)))
                #signals = [file.readsignal(i) for i in range(signalsNum)]

            if path.endswith('.nii.gz') or path.endswith('.nii'):
                print('NII')
                #file = load(path)
                #data = file.get_fdata()

            if path.endswith('.mat'):
                print("MAT")
                file = loadmat(path)
                data_key = list(file.keys())[-1]
                data = file[data_key]
                print(type(data))

            if path.endswith('.csv'):
                print("CSV")
                data = read_csv(path)

            dataType = self.classifyData(data)
            #model = self.getModel(dataType)
            #results = self.processData(data, model)
            self.showResult(data)

    def classifyData(self, data):
        print(data.shape)
        return "EEG"
        return "MRI/CNN"

    def getModel(self,type):
        #model = load_model(rf'{type}/MODEL/0.7974.h5')
        #return model
        pass

    # def processData(self, DATA, model):
    #
    #     DATA_FILTERED = filterEEGData(DATA)
    #
    #     DATA_CLIPPED = clipEEGData(DATA_FILTERED)
    #
    #     DATA_NORMALIZED = normalizeEEGData(DATA_CLIPPED)
    #
    #     DATA_FRAMED = frameDATA(DATA_NORMALIZED)
    #
    #     predictions = model.predict(DATA_FRAMED)
    #
    #     checkResult(predictions)
    #
    #     return predictions

    def showResult(self,data):
        print("plot")
        print("predicts")


# 1. Wprowadzenie danych

    # obsługa wielu plików naraz (np. lekarz wrzuca naraz 15 eeg i 10 mri)

    # obsługa mri i eeg naraz
        # analiza struktury wprowadzonych danych
        # na tej podstawie podjecie decyzji czy zostalo wprowadzone eeg czy mri

    # obsługa eeg o różnych ilościach kanałów
        # analiza struktury wprowadzonych danych
        # na tej podstawie wybranie modelu dostosowanego do konkretnej ilosci kanałów

    # obsługa różnych płaszczyzn mri (różne modele nauczone na różnych płaszczyznach)

# 2. Wyświetlenie diagnozy

    # diagnoza od razu dla wszystkich wprowadzonych danych

    # wyswietlenie danych na których podstawie zostala postawiona diagnoza (wykresy EEG / zdjecia MRI)
        # dla większej ilości danych możliwość przewijania zdjęć

    # wspolna diagnoza dla roznych danych tego samego pacjenta

# Ponadto

    # dodanie do modelu MRI etykiety dotyczącej płaszczyzny mózgu na której uczony był model
    # dodanie do modelu etykiet dotyczących charakterystyki grupy uczącej (np. wiek, płeć, dominująca ręka)