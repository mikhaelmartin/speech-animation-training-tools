import os
import sys
import math
import moviepy.editor as mp

import numpy as np
import librosa
import librosa.display

"""
    James Lyons et al. (2020, January 14).
    jameslyons/python_speech_features: release v0.6.1 (Version 0.6.1). Zenodo.
    http://doi.org/10.5281/zenodo.3607820
    source: https://github.com/jameslyons/python_speech_features
"""
import python_speech_features as psf
import scipy.io.wavfile as wav

# from scipy.signal.windows import hann
import csv
import subprocess


def wav_extract(source_file_path, output_folder, fps=44100):
    audioclip = mp.AudioFileClip(source_file_path, fps=fps)
    audioclip.write_audiofile(
        output_folder + source_file_path.split("/")[-1].split(".")[-2] + ".wav"
    )


def get_fps(source_file_path):
    clip = mp.VideoFileClip(source_file_path)
    return clip.fps


def psf_mfcc_extract(
    source_file_path,
    output_folder,
    fft_window_size=2048,
    hop_length=int(44100 / 25),
    n_mfcc=13,
    n_mels=40,
):
    rate, signal = wav.read(source_file_path)
    MFCCs = psf.mfcc(
        signal,
        samplerate=rate,
        winlen=fft_window_size / rate,
        winstep=hop_length * 2 / rate,  # hop_length / rate,
        nfft=fft_window_size,
        numcep=n_mfcc,
        nfilt=n_mels,
        preemph=0,
        ceplifter=0,
        appendEnergy=False,
        winfunc=np.hanning,
    ).T
    myFile = open(
        output_folder
        + source_file_path.split("/")[-1].split(".")[-2]
        + "_mfcc.csv",
        "w",
    )
    writer = csv.writer(myFile)
    writer.writerow(["mfcc_" + str(i + 1) for i in range(13)])
    writer.writerows(MFCCs.transpose())


def librosa_mfcc_extract(
    source_file_path,
    output_folder,
    sample_rate=44100,
    fft_window_size=2048,
    hop_length=int(44100 / 25),
    n_mfcc=13,
    n_mels=40,
):
    signal, sr = librosa.load(source_file_path, sr=sample_rate)
    MFCCs = librosa.feature.mfcc(
        signal,
        sr=sr,
        n_fft=fft_window_size,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        htk=False,
    )
    myFile = open(
        output_folder
        + source_file_path.split("/")[-1].split(".")[-2]
        + "_mfcc.csv",
        "w",
    )
    writer = csv.writer(myFile)
    writer.writerow(["mfcc_" + str(i + 1) for i in range(13)])
    writer.writerows(MFCCs.transpose())


def decrease_frame(file, lines, frame_count, target_frame_count):
    difference = frame_count - target_frame_count
    for i in range(difference):
        lines.pop(-1)
    for line in lines:
        file.write(line)
    file.close()


def equate_frame_count(filenames):
    # ambil lines dari setiap file
    files_lines = [open(filename).readlines() for filename in filenames]
    # ambil jumlah frame dari setiap file
    frame_counts = [(len(lines) - 1) for lines in files_lines]
    # dapetin jumlah frame paling sedikit dari file2 tsb
    min_frame_count = sys.maxsize
    for frame_count in frame_counts:
        if frame_count < min_frame_count:
            min_frame_count = frame_count
    # print("min frame count:", min_frame_count)
    # untuk setiap file yang framenya kelebihan. kurangi hingga sama
    for i in range(len(filenames)):
        decrease_frame(
            open(filenames[i], "w", newline=""),
            files_lines[i],
            frame_counts[i],
            min_frame_count,
        )


def set_speaker_face_id(file_path, id):
    csv_reader = csv.reader(open(file_path), delimiter=",")
    lines = list(csv_reader)
    csv_writer = csv.writer(open(file_path, "w", newline=""), delimiter=",")
    csv_writer.writerow(lines[0])
    face_id_index = lines[0].index("face_id")
    for line in lines[1:]:
        line[face_id_index] = id
        csv_writer.writerow(line)


video_dir = "../video/"
features_dir = "../features/"
if not os.path.exists(features_dir):
    os.mkdir(features_dir)

# video is put inside folder for each speaker
speaker_dir_list = [
    file for file in os.listdir(video_dir) if os.path.isdir(video_dir + file)
]
speaker_dir_list.sort()

for i, speaker in enumerate(speaker_dir_list):
    print("getting features for " + speaker)
    speaker_video_dir = video_dir + speaker + "/"
    speaker_feature_dir = features_dir + speaker
    audio_dir = features_dir + speaker + "/audio/"
    mfcc_dir = features_dir + speaker + "/mfcc/"
    openface_dir = features_dir + speaker + "/openface/"
    if not os.path.exists(speaker_feature_dir):
        os.mkdir(speaker_feature_dir)
    if not os.path.exists(audio_dir):
        os.mkdir(audio_dir)
    if not os.path.exists(mfcc_dir):
        os.mkdir(mfcc_dir)
    if not os.path.exists(openface_dir):
        os.mkdir(openface_dir)

    print("getting video name list")
    video_name_list = [
        item
        for item in os.listdir(video_dir + speaker)
        if (".mp4" in item)
        or (".wmv" in item)
        or (".avi" in item)
        or (".webm" in item)
        or (".mpg" in item)
    ]

    print("getting fps list")
    fps_list = [get_fps(speaker_video_dir + item) for item in video_name_list]

    print("create data list")
    # ambil nama data. filename tanpa ekstensi
    data_list = [item.split(".")[-2] for item in video_name_list]

    print("getting landmark data")
    # dari video ambil landmark taro di folder openface_output
    for item in video_name_list:
        subprocess.run(
            [
                "/home/alkhemi/OpenFace/build/bin/FeatureExtraction",
                "-3Dfp",
                "-pose",
                # "-tracked",
                "-f",
                speaker_video_dir + item,
                "-out_dir",
                openface_dir,
            ]
        )

    # dari video ambil audio taro ke folder audio
    for item in video_name_list:
        wav_extract(speaker_video_dir + item, audio_dir)

    print("getting audio name list")
    audio_name_list = [
        item for item in os.listdir(audio_dir) if (".wav" in item)
    ]

    print("getting MFCCs")
    # dari audio ambil mfcc taro di folder mfcc
    for index in range(len(audio_name_list)):
        psf_mfcc_extract(
            audio_dir + audio_name_list[index],
            mfcc_dir,
            hop_length=math.floor(44100 / fps_list[index]),
        )

    for data in data_list:
        # cek jumlah frame
        # kalau ada perbedaan samain dengan yang jumlah framenya lebih kecil.
        equate_frame_count(
            [mfcc_dir + data + "_mfcc.csv", openface_dir + data + ".csv"]
        )
        set_speaker_face_id(openface_dir + data + ".csv", i)
