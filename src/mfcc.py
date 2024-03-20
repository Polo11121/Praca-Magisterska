import librosa
from src.audio_dct import generate_the_cepstral_coefficents
from src.audio_fft import apply_fft
from src.audio_framing import frame_audio
from src.audio_log import log_audio
from src.audio_mel_scale_filter_bank import apply_mel_scale_filter_bank
from src.audio_preprocessing import preprocess_audio
from src.audio_signal_power import calculate_signal_power
from src.audio_windowing import aplly_window
from src.config import PRE_EMPHASIS_COEFFICIENT, FFT_SIZE, HOP_SIZE, DCT_FILTER_NUM, MEL_FILTER_NUM


def mfcc(audio_path, FFT_size=FFT_SIZE, hop_size=HOP_SIZE, mel_filter_num=MEL_FILTER_NUM, dct_filter_num=DCT_FILTER_NUM,
         pre_emphasis_coefficient=PRE_EMPHASIS_COEFFICIENT):
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')

    audio_preprocess = preprocess_audio(audio, pre_emphasis_coefficient)

    audio_framed = frame_audio(audio_preprocess, FFT_size, hop_size, sample_rate)

    audio_windowed = aplly_window(audio_framed, FFT_size)

    audio_fft = apply_fft(audio_windowed, FFT_size)

    audio_power = calculate_signal_power(audio_fft)

    audio_mel_scale_filter_bank = apply_mel_scale_filter_bank(sample_rate, FFT_size, mel_filter_num, audio_power)

    audio_log = log_audio(audio_mel_scale_filter_bank)

    cepstral_coefficients = generate_the_cepstral_coefficents(dct_filter_num, mel_filter_num, audio_log)

    return cepstral_coefficients
