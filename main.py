from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
import sys
from MRI.main_MRI import MRI
from EEG.main_EEG import EEG
import os

from CONTROLLERS.DoctorViewController import DoctorViewController

current_dir = os.path.dirname(__file__)
UI_PATH = rf'{current_dir}/UI'

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #self.run_app()
        self.viewController = None
        self.loadDoctorUI()
        self.show()

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

    def loadDoctorUI(self):
        self.viewController = DoctorViewController(self)
        self.viewController.ui.switchSceneBtn.clicked.connect(self.loadAdminUI)

    def loadAdminUI(self):
        print("admin view")
        # self.viewController = AdminViewController(self)
        # self.viewController.ui.switchSceneBtn.clicked.connect(self.loadDoctorUI)

    def runEEG(self):
        EEG()

    def runMRI(self):
        MRI()

app = QApplication(sys.argv)
window = MainWindow()
app.exec_()

