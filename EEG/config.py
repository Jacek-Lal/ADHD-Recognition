#SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
EEG_SUBFOLDERS = ['ADHD', 'CONTROL']                                                # Wszystkie podfoldery z głównego folderu z danymi
EEG_POS_PHRASE = 'ADHD'                                                             # Fraza zawierająca się w nazwie folderu z grupą chorych pacjentów
EEG_NEG_PHRASE = 'CONTROL'                                                          # Fraza zawierająca się w nazwie folderu z grupą kontrolną
EEG_SIGNAL_FRAME_SIZE = 128                                                        # wielkość pojedyńczej próbki pobieranej z danych elektrody
#SPRAWDZ TĄ ŚCIEŻKĘ I POPRAW WZGLĘDNĄ
CNN_POS_LABEL = 1                                                             # wyjście pozytywne z modelu
CNN_NEG_LABEL = 0                                                           # wyjście negatywne
CNN_INPUT_SHAPE = (19, EEG_SIGNAL_FRAME_SIZE, 1)                                    # 19 - ilość elektrod ; wielkość ramki danych ; 1 - magiczna liczba którą wymaga keras
CNN_EPOCHS = 1                                                                    # Ilość przejść do przodu przez model
CUTOFFS = [(4,8), (13,30), (4,30)]                                                  # Przedziały częstotliwości theta, beta, wszystko
FS = 128                                                                            # Częstotliwośc próbkowania