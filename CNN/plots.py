import numpy as np
import matplotlib.pyplot as plt
from config import CUTOFFS
def plot(data, patient_numer, channel_number):
    fs = 128
    t = np.arange(0, data[patient_numer][channel_number].shape[0])/fs
    plt.plot(t, data[patient_numer][channel_number])
    print(f"Czas dla kanału: {t[-1]} s.")
    print(f"Ilość próbek: {data[patient_numer][channel_number].shape}")

def plot_frequency_band(data, fs, band_number):
    # Transformata Fouriera
    frequencies = np.fft.fftfreq(len(data), d=1/fs)
    fft_values = np.fft.fft(data)
    magnitude_spectrum = np.abs(fft_values)

    # Wykres widma częstotliwościowego
    plt.plot(frequencies, magnitude_spectrum, label=f'{CUTOFFS[band_number]} Hz')
    plt.title('Widmo częstotliwościowe')
    plt.xlabel('Częstotliwość (Hz)')
    plt.ylabel('Amplituda')
    plt.legend()