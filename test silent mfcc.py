import os
import numpy as np
import multiprocessing
import librosa
import csv
import time

frame_length = 1 / 60  # 0.02  # 20ms
audio_sample_rate = 44100
hop_length = int(audio_sample_rate * frame_length)  # 735 frames for 16.66 ms

audio_dir = "./"
features_dir = "../features/"
if not os.path.exists(features_dir):
    os.mkdir(features_dir)


def librosa_mfcc_extract():
    sr = 44100
    signal = np.zeros((44100,))
    MFCCs = librosa.feature.mfcc(
        signal,
        sr=sr,
        n_fft=2048,  # window size around 46.44ms
        hop_length=hop_length,
        n_mfcc=13,  # num of coefficients
        n_mels=40,  # num of mel band filters
        window="hann",
        htk=True,
    )
    # delta features from librosa are computed Savitsky-Golay filtering
    delta_MFCCs = librosa.feature.delta(MFCCs)
    delta2_MFCCs = librosa.feature.delta(MFCCs, order=2)
    MFCCs = MFCCs.transpose()
    delta_MFCCs = delta_MFCCs.transpose()
    delta2_MFCCs = delta2_MFCCs.transpose()
    myFile = open(
        audio_dir+"silent_mfcc.csv",
        "w",
    )
    writer = csv.writer(myFile)
    writer.writerow(
        ["frame", "timestamp"]
        + ["mfcc_" + str(i + 1) for i in range(13)]
        + ["delta_mfcc_" + str(i + 1) for i in range(13)]
        + ["delta2_mfcc_" + str(i + 1) for i in range(13)]
    )
    # face_id = int(speaker.split("s")[-1])
    for i in range(len(MFCCs)-1):
        writer.writerow(
            [i + 1]
            # + [face_id]
            + [np.round(i * frame_length, 3)]
            + [np.round(x, 3) for x in MFCCs[i].tolist()]
            + [np.round(x, 3) for x in delta_MFCCs[i].tolist()]
            + [np.round(x, 3) for x in delta2_MFCCs[i].tolist()]
        )
    myFile.close()


librosa_mfcc_extract()


