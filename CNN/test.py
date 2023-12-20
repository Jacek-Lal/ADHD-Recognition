from eeg_read import *
from config import *
from plots import *
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
import copy



ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)

ADHD_1HZ, CONTROL_1HZ = highpassFilterEEG(ADHD_DATA, CONTROL_DATA)



# Przykład użycia
# Wynik będzie zawierał wektory danych EEG dla każdego pacjenta
# Dla każdego pacjenta, wartości z poszczególnych kanałów są dodawane do nowego wektora
nowa_tablica = filterICA(ADHD_DATA)

print(nowa_tablica.shape)


    #info = mne.create_info(ch_names=CHANNELS, sfreq=FS, ch_types='eeg')


    #raw = mne.io.RawArray(ADHD_reshaped, info)

    # Wyświetlenie informacji o obiekcie Raw
    #print(raw.info)

    # Wizualizacja danych
    #raw.plot()











plt.show()