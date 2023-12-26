import numpy as np
import matplotlib.pyplot as plt
from CNN.config import *

def plot(data, patient_number, channel_number):

    t = np.arange(0, data[patient_number][channel_number].shape[0]) / FS
    signal = data[patient_number][channel_number]
    print(f"Ilość próbek: {data[patient_number][channel_number].shape[0]}")
    print(f"Czas: {t[-1]:.3f} s")
    plt.plot(t, signal, label=f'Pacjent {patient_number}, Kanał {channel_number}')
    plt.xlabel('Czas (s)')
    plt.ylabel('Wartości próbek')
    plt.title('Wykres sygnału EEG')
    plt.legend()
    plt.show()

def plot_frequency_band(data, band_number):
    #data = wektor pojedynczy

    frequencies = np.fft.fftfreq(len(data), d=1/FS)
    fft_values = np.fft.fft(data)
    magnitude_spectrum = np.abs(fft_values)

    plt.plot(frequencies, magnitude_spectrum, label=f'{CUTOFFS[band_number]} Hz')
    plt.title('Widmo częstotliwościowe')
    plt.xlabel('Częstotliwość (Hz)')
    plt.ylabel('Amplituda')
    plt.legend()

def plot_with_treshold(data, patient_number, channel_number, treshold):
    signal_length = data[patient_number][channel_number].shape[0]

    plot(data, patient_number, channel_number)
    plt.plot([x/FS for x in range(signal_length)], [treshold for x in range(signal_length)])
    plt.plot([x/FS for x in range(signal_length)], [-treshold for x in range(signal_length)])
    plt.xlabel('Czas (s)')
    plt.ylabel('Wartości próbek')
    plt.title('Wykres sygnału po zastosowaniu tresholda')
    plt.legend()