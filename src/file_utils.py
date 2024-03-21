import fnmatch
import os
import numpy as np
from src.config import FFT_SIZE, HOP_SIZE, DCT_FILTER_NUM, MEL_FILTER_NUM
from src.mfcc import mfcc


def get_mfcc_filename(audio_path, FFT_size, hop_size, mel_filter_num, dct_filter_num):
    file_name = os.path.splitext(os.path.basename(audio_path))[0]

    return f"dataSources/mfccs/{file_name}_FFT{FFT_size}_HOP{hop_size}_MEL{mel_filter_num}_DCT{dct_filter_num}.npy"


def save_mfcc(audio_path, mfccs_features, FFT_size, hop_size, mel_filter_num, dct_filter_num):
    file_path = get_mfcc_filename(audio_path, FFT_size, hop_size, mel_filter_num, dct_filter_num)
    np.save(file_path, mfccs_features)


def load_mfcc(audio_path, FFT_size, hop_size, mel_filter_num, dct_filter_num):
    file_path = get_mfcc_filename(audio_path, FFT_size, hop_size, mel_filter_num, dct_filter_num)
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        return None


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def features_extractor(audio_path, FFT_size=FFT_SIZE, hop_size=HOP_SIZE, mel_filter_num=MEL_FILTER_NUM,
                       dct_filter_num=DCT_FILTER_NUM):
    mfccs_features = load_mfcc(audio_path, FFT_size, hop_size, mel_filter_num, dct_filter_num)
    if mfccs_features is None:
        mfccs_features = mfcc(audio_path, FFT_size, hop_size, mel_filter_num, dct_filter_num)
        save_mfcc(audio_path, mfccs_features, FFT_size, hop_size, mel_filter_num, dct_filter_num)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features
