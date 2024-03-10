#from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QGridLayout, QLineEdit, QComboBox
#from PyQt5.QtGui import QTextDocument, QRegExpValidator
#from PyQt5.QtCore import Qt, QRegExp
#from PyQt5.QtXml import QDomElement
#from PyQt5 import uic
#import sys
from MRI.main_MRI import MRI
from EEG.main_EEG import EEG
import os

#current_dir = os.path.dirname(__file__)
#UI_PATH = rf'{current_dir}/UI'

#eee
#aaa
class MainWindow():
    def __init__(self):
        super().__init__()
        self.run_app()
        #self.loadMainUI()
        #self.show()
    def run_app(self):
        while True:
            try:
                choice = input('Wybierz opcję:   1-(EEG)   2-(MRI): ')
                if choice == '1':
                    self.runEEG()
                    break
                elif choice == '2':
                    self.runMRI()
                    break
                else:
                    print("Niepoprawny wybór. Wprowadź 1 lub 2.")
            except Exception as e:
                print(f"Wystąpił błąd: {e}")

    #def loadMainUI(self):
    #    uic.loadUi(rf'{UI_PATH}/MainWindow.ui', self)
    #    self.btn_EEG.clicked.connect(self.runEEG)
    #    self.btn_MRI.clicked.connect(self.runMRI)

    def runEEG(self):
        EEG()

    def runMRI(self):
        MRI()

#app = QApplication(sys.argv)
window = MainWindow()
#app.exec_()

