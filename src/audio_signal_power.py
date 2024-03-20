import numpy as np


def calculate_signal_power(audio_fft):
    audio_power = np.square(np.abs(audio_fft))

    return audio_power
