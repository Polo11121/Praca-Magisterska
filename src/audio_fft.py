import numpy as np
import scipy.fftpack as fft


def custom_fft(audio, FFT_size):
    audio_T = np.transpose(audio)
    audio_fft = np.empty((int(1 + FFT_size // 2), audio_T.shape[1]), dtype=np.complex64, order='F')

    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_T[:, n], axis=0)[:audio_fft.shape[0]]

    audio_fft = np.transpose(audio_fft)

    return audio_fft


def apply_fft(audio, FFT_size):
    audio_fft = custom_fft(audio, FFT_size)

    return audio_fft
