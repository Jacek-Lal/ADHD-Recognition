import numpy as np
import matplotlib.pyplot as plt
from config import CUTOFFS, FS

def plot(data, patient_numer, channel_number):
    t = np.arange(0, data[patient_numer][channel_number].shape[0])/FS
    plt.plot(t, data[patient_numer][channel_number])
    print(f"Czas dla kanału: {t[-1]:.3f} s.")
    print(f"Ilość próbek: {data[patient_numer][channel_number].shape[0]}")

def plot_frequency_band(data, band_number):

    frequencies = np.fft.fftfreq(len(data), d=1/FS)
    fft_values = np.fft.fft(data)
    magnitude_spectrum = np.abs(fft_values)

    plt.plot(frequencies, magnitude_spectrum, label=f'{CUTOFFS[band_number]} Hz')
    plt.title('Widmo częstotliwościowe')
    plt.xlabel('Częstotliwość (Hz)')
    plt.ylabel('Amplituda')
    plt.legend()
