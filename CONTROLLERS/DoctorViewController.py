from PyQt5 import uic
import os

from PyQt5.QtWidgets import QFileDialog
from pyedflib import EdfReader
from nibabel import load
from scipy.io import loadmat
from pandas import read_csv

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
        self.filePaths, _ = QFileDialog.getOpenFileNames(self.mainWindow, "Choose files", "", "", options=options)
        self.processFiles()

    def processFiles(self):

        for path in self.filePaths:

            # załaduj plik ( zależnie od typu inaczej, po to tyle if'ów)
            # zdecyduj czy eeg czy mri  (na podstawie struktury/rozszerzenia)
            # wybierz z niego potrzebne dane
            # wybierz model na podstawie struktury danych (np dla EEG różna ilość kanałów, dla mri różna płaszczyzna mózgu)
            # wrzuć dane w model
            # zwróc wynik

            if path.endswith('.edf'):
                print("EDF")
                file = EdfReader(path)
                signalsNum = file.signals_in_file
                signals = [file.readsignal(i) for i in range(signalsNum)]

            if path.endswith('.nii.gz') or path.endswith('.nii'):
                print('NII')
                file = load(path)
                data = file.get_fdata()

            if path.endswith('.mat'):
                print("MAT")
                data = loadmat(path)

            if path.endswith('.csv'):
                print("CSV")
                data = read_csv(path)

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