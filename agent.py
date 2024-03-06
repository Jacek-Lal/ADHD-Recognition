from PyQt5 import QtWidgets, uic
import EEG.config
import subprocess
import sys


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('UI_projekt.ui', self)
        self.show()

        sys.stdout = self

        self.pushButton_5.clicked.connect(self.train_cnn)

    def write(self, text):
        self.textEdit_3.append(text)

    def train_cnn(self):
        epochs = int(self.textEdit.toPlainText())
        batch_size = int(self.textEdit_2.toPlainText())

        EEG.config.set_cnn_epochs(epochs)
        EEG.config.set_cnn_batch_size(batch_size)

        print("Updated CNN_EPOCHS:", EEG.config.CNN_EPOCHS)
        print("Updated CNN_BATCH_SIZE:", EEG.config.CNN_BATCH_SIZE)

        script = 'EEG/main_EEG.py'
        subprocess.run(['python', script])

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    app.exec_()