from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QGridLayout, QLineEdit, QComboBox
from PyQt5.QtGui import QTextDocument, QRegExpValidator
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtXml import QDomElement
from PyQt5 import uic
import sys
from MRI.main_MRI import MRI
from EEG.main_EEG import EEG
import os

current_dir = os.path.dirname(__file__)
UI_PATH = rf'{current_dir}/UI'


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.loadMainUI()
        self.show()

    def loadMainUI(self):
        uic.loadUi(rf'{UI_PATH}/MainWindow.ui', self)
        self.btn_EEG.clicked.connect(self.runEEG)
        self.btn_MRI.clicked.connect(self.runMRI)

    def runEEG(self):
        self.close()  # wystarczyło dodać zamykanie okna a nie komentowac pol kodu
        print("Uruchomienie EEG")
        EEG()

    def runMRI(self):
        self.close()
        print("Uruchomienie MRI")
        MRI()

app = QApplication(sys.argv)
window = MainWindow()
app.exec_()


