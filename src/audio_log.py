import numpy as np


def log_audio(audio):
    audio_log = 10.0 * np.log10(audio + 1e-10)

    return audio_log
