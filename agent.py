from PyQt5 import QtWidgets, uic
import EEG.config
import subprocess
import sys


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('aUI_projekt_EEG.ui', self)
        self.show()

        sys.stdout = self

        self.startButton.clicked.connect(self.train_cnn)

    def write(self, text):
        self.terminal_Window.append(text)

    def train_cnn(self):
        epochs = int(self.textEdit_epochs.toPlainText()) if self.textEdit.toPlainText().strip() else EEG.config.CNN_EPOCHS
        batch_size = int(self.textEdit_batch_size.toPlainText()) if self.textEdit_batch_size.toPlainText().strip else EEG.config.CNN_BATCH_SIZE
        learning_rate = int(self.textEdit_learning_rate.toPlainText()) if self.textEdit_learning_rate.toPlainText().strip else EEG.config.CNN_LEARNING_RATE
        electrodes = int(self.textEdit_electrodes.toPlainText()) if self.textEdit_electrodes.toPlainText().strip else EEG.config.EEG_NUM_OF_ELECTRODES
        frame_size = int(self.textEdit_frame_size.toPlainText()) if self.textEdit_frame_size.toPlainText().strip else EEG.config.EEG_SIGNAL_FRAME_SIZE
        frequency = int(self.textEdit_frequency.toPlainText()) if self.textEdit_frequency.toPlainText().strip else EEG.config.FS
        test_size = int(self.textEdit_test_size.toPlainText()) if self.textEdit_test_size.toPlainText().strip else EEG.config.CNN_TEST_SIZE

        EEG.config.set_cnn_epochs(epochs)
        EEG.config.set_cnn_batch_size(batch_size)
        EEG.config.set_learning_rate(learning_rate)
        EEG.config.set_electrodes(electrodes)
        EEG.config.set_frame(frame_size)
        EEG.config.set_fs(frequency)
        EEG.config.set_test_size(test_size)

        print("CNN_EPOCHS:", EEG.config.CNN_EPOCHS)
        print("CNN_BATCH_SIZE:", EEG.config.CNN_BATCH_SIZE)
        print("CNN_LEARNING_RATE:", EEG.config.CNN_LEARNING_RATE)
        print("CNN_TEST_SIZE:", EEG.config.CNN_TEST_SIZE)
        print("EEG_NUM_OF_ELECTRODES:", EEG.config.EEG_NUM_OF_ELECTRODES)
        print("EEG_SIGNAL_FRAME_SIZE:", EEG.config.EEG_SIGNAL_FRAME_SIZE)
        print("FS:", EEG.config.FS)

        script = 'EEG/main_EEG.py'
        subprocess.run(['python', script])

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    app.exec_()