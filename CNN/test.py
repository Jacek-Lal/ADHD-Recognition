from eeg_read import *
from config import *
from plots import *

ADHD_DATA, CONTROL_DATA = readEEGRaw(EEG_DATA_PATH)
ADHD_NORMALIZED, CONTROL_NORMALIZED = normalizeEEGData(ADHD_DATA, CONTROL_DATA)
ADHD_FILTERED, CONTROL_FILTERED = filterEEGData(ADHD_NORMALIZED, CONTROL_NORMALIZED)

bandwith = 0
patient = 10
channel = 10

plot(ADHD_DATA, patient, channel)
plot(ADHD_FILTERED[bandwith], patient, channel)
plt.show()

# Przykładowy sygnał w dziedzinie czasu
T = 1/FREQ   # Okres próbkowania
t = np.arange(0, ADHD_FILTERED[bandwith][patient][channel].shape[0])/FREQ

# Wykonaj FFT
fft_result = np.fft.fft(ADHD_FILTERED[bandwith][patient][channel])
frequencies = np.fft.fftfreq(len(fft_result), T)  # Wektor częstotliwości

for i, data in enumerate(ADHD_FILTERED):
    # Wykres widma częstotliwościowego dla danego pasma
    plot_frequency_band(data[patient][channel], i)
    
plt.show()
