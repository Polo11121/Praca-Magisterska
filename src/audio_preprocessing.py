import numpy as np


def normalize_audio(audio):
    audio_normalized = audio / np.max(np.abs(audio))

    return audio_normalized


def apply_pre_emphasis(audio, alpha):
    audio_emphasized = np.append(audio[0], audio[1:] - alpha * audio[:-1])

    return audio_emphasized


def preprocess_audio(audio, alpha):
    audio_normalized = normalize_audio(audio)

    audio_normalized_emphasized = apply_pre_emphasis(audio_normalized, alpha)

    return audio_normalized_emphasized
