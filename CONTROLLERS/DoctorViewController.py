from PyQt5 import uic
import os

current_dir = os.path.dirname(__file__)
UI_PATH = rf'{current_dir}/UI'
parent_directory = os.path.dirname(current_dir)

class DoctorViewController:
    def __init__(self, mainWindow):
        self.ui = uic.loadUi(rf'{parent_directory}/UI/doctorView.ui', mainWindow)
        self.addEvents()
    def addEvents(self):
        self.ui.loadDataBtn.clicked.connect(self.loadData)

    def loadData(self):
        print("chuj")

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