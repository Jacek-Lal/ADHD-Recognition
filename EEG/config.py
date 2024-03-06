#SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
EEG_DATA_PATH = 'EEG/TRAIN/TRAIN_DATA'                                             # Folder z danymi
EEG_SUBFOLDERS = ['ADHD', 'CONTROL']                                                # Wszystkie podfoldery z głównego folderu z danymi
EEG_POS_PHRASE = 'ADHD'                                                             # Fraza zawierająca się w nazwie folderu z grupą chorych pacjentów
EEG_NEG_PHRASE = 'CONTROL'                                                          # Fraza zawierająca się w nazwie folderu z grupą kontrolną
EEG_SIGNAL_FRAME_SIZE = 128                                                        # wielkość pojedyńczej próbki pobieranej z danych elektrody
#SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
CNN_MODELS_PATH = '../EEG/MODEL'                                                       # folder do zapisywania modeli
CNN_POS_LABEL = 1                                                             # wyjście pozytywne z modelu
CNN_NEG_LABEL = 0                                                           # wyjście negatywne
CNN_INPUT_SHAPE = (19, EEG_SIGNAL_FRAME_SIZE, 1)                                    # 19 - ilość elektrod ; wielkość ramki danych ; 1 - magiczna liczba którą wymaga keras
CNN_EPOCHS = 1                                                                    # Ilość przejść do przodu przez model
CNN_BATCH_SIZE = 32
CUTOFFS = [(4,8), (13,30), (4,30)]                                                  # Przedziały częstotliwości theta, beta, wszystko
FS = 128                                                                            # Częstotliwośc próbkowania

def set_cnn_epochs(new_value):
    global CNN_EPOCHS
    CNN_EPOCHS = new_value

def set_cnn_batch_size(new_value):
    global CNN_BATCH_SIZE
    CNN_BATCH_SIZE = new_value
