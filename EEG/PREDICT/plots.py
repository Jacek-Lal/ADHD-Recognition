import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add the directory containing config.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from EEG.config import *

def plot(data, channel_number):

    t = np.arange(0, data[channel_number].shape[0]) / FS

    signal = data[channel_number]

    print(f"Ilość próbek: {data[channel_number].shape[0]}")
    print(f"Czas: {t[-1]:.3f} s")

    plt.plot(t, signal, label=f'Kanał {channel_number}')

    plt.xlabel('Czas (s)')
    plt.ylabel('Wartości próbek')
    plt.title('Wykres sygnału TRAIN_DATA')

    plt.legend()
    plt.show()

def plot_frequency_band(data, channel_number, band_number=2):
    # data = dane z kanału

    frequencies = np.fft.fftfreq(len(data[channel_number]), d=1/FS)
    fft_values = np.fft.fft(data[channel_number])
    magnitude_spectrum = np.abs(fft_values)

    plt.plot(frequencies, magnitude_spectrum, label=f'{CUTOFFS[band_number]} Hz')

    plt.title('Widmo częstotliwościowe')
    plt.xlabel('Częstotliwość (Hz)')
    plt.ylabel('Amplituda')

    plt.legend()
    plt.show()