import numpy as np
import librosa
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


# bikin filter
sos = butter(13, [60, 8000], output="sos", btype="bandpass", fs=44100)


def GetMFCCsFromAudio(
    audio_file_path, n_fft=2048, n_mfcc=13, n_mels=40, window="hann", htk=True
):
    # filter audio
    rate, audio_array = wavfile.read(audio_file_path)
    audio_array = sosfilt(sos, audio_array)

    frame_length = 1 / 60  # 16.66
    hop_length = int(rate * frame_length)  # 735 frames @ 44.1kHz

    MFCCs = librosa.feature.mfcc(
        audio_array,
        sr=rate,
        n_fft=n_fft,  # window size 2048 is around 46.44ms @ 44.1kHz
        hop_length=hop_length,
        n_mfcc=n_mfcc,  # num of coefficients
        n_mels=n_mels,  # num of mel band filters
        window=window,
        htk=htk,
    )
    delta_MFCCs = librosa.feature.delta(MFCCs)
    delta2_MFCCs = librosa.feature.delta(MFCCs, order=2)
    return np.vstack((MFCCs, delta_MFCCs, delta2_MFCCs)).transpose()