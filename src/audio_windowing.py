import numpy as np


def get_window(N, fftbins=True):
    if fftbins:
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))
    else:
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(0.5, N - 0.5) / N))


def aplly_window(audio_framed, FFT_size):
    window = get_window(FFT_size, fftbins=True)

    audio_windowed = audio_framed * window

    return audio_windowed
