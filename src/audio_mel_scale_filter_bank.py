import numpy as np


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_freq(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
    freqs = mel_to_freq(mels)

    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs


def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points) - 2, int(FFT_size / 2 + 1)))

    for n in range(len(filter_points) - 2):
        filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[
            n + 1])

    return filters


def mel_scale_filter_bank(sample_rate, FFT_size, mel_filter_num):
    freq_min = 0
    freq_high = sample_rate / 2
    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate)
    filters = get_filters(filter_points, FFT_size)
    enorm = 2.0 / (mel_freqs[2:mel_filter_num + 2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]

    return filters


def apply_mel_scale_filter_bank(sample_rate, FFT_size, mel_filter_num, audio):
    filters = mel_scale_filter_bank(sample_rate, FFT_size, mel_filter_num)
    audio_mel_scale_filter_bank = np.dot(filters, np.transpose(audio))

    return audio_mel_scale_filter_bank
