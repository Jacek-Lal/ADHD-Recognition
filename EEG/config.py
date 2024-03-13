#SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
EEG_DATA_PATH = 'EEG/TRAIN/TRAIN_DATA'                                             # Folder z danymi
EEG_SUBFOLDERS = ['ADHD', 'CONTROL']                                                # Wszystkie podfoldery z głównego folderu z danymi
EEG_POS_PHRASE = 'ADHD'                                                             # Fraza zawierająca się w nazwie folderu z grupą chorych pacjentów
EEG_NEG_PHRASE = 'CONTROL'                                                          # Fraza zawierająca się w nazwie folderu z grupą kontrolną
EEG_SIGNAL_FRAME_SIZE = 128                                                        # wielkość pojedyńczej próbki pobieranej z danych elektrody
EEG_NUM_OF_ELECTRODES = 19
CNN_MODELS_PATH = '../EEG/MODEL'                                                       # folder do zapisywania modeli
CNN_POS_LABEL = 1                                                             # wyjście pozytywne z modelu
CNN_NEG_LABEL = 0                                                           # wyjście negatywne
CNN_INPUT_SHAPE = (EEG_NUM_OF_ELECTRODES, EEG_SIGNAL_FRAME_SIZE, 1)                                    # 19 - ilość elektrod ; wielkość ramki danych ; 1 - magiczna liczba którą wymaga keras
CNN_EPOCHS = 1                                                                    # Ilość przejść do przodu przez model
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.0001
CNN_TEST_SIZE = 0.2
CUTOFFS = [(4,8), (13,30), (4,30)]                                                  # Przedziały częstotliwości theta, beta, wszystko
FS = 128                                                                            # Częstotliwośc próbkowania

def set_cnn_epochs(new_value):
    global CNN_EPOCHS
    CNN_EPOCHS = new_value

def set_cnn_batch_size(new_value):
    global CNN_BATCH_SIZE
    CNN_BATCH_SIZE = new_value

def set_learning_rate(new_value):
    global CNN_LEARNING_RATE
    CNN_LEARNING_RATE = new_value

def set_electrodes(new_value):
    global EEG_NUM_OF_ELECTRODES
    EEG_NUM_OF_ELECTRODES = new_value

def set_frame(new_value):
    global EEG_SIGNAL_FRAME_SIZE
    EEG_SIGNAL_FRAME_SIZE = new_value

def set_fs(new_value):
    global FS
    FS = new_value

def set_test_size(new_value):
    global CNN_TEST_SIZE
    CNN_TEST_SIZE = new_value