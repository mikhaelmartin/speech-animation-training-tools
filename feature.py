import os

import moviepy.editor as mp

import numpy as np
import librosa
import librosa.display
import csv
import subprocess


def wav_extract(source_file_path, output_folder, fps=44100):
    audioclip = mp.AudioFileClip(source_file_path, fps=fps)
    audioclip.write_audiofile(
        output_folder + source_file_path.split("/")[-1].split(".")[-2] + ".wav"
    )


def mfcc_extract(
    source_file_path,
    output_folder,
    sample_rate=44100,
    fft_window_size=2048,
    hop_length=int(44100 / 25),
    n_mfcc=13,
):
    signal, sr = librosa.load(source_file_path, sr=sample_rate)
    MFCCs = librosa.feature.mfcc(
        signal,
        sr=sr,
        n_fft=fft_window_size,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
    )
    myFile = open(
        output_folder
        + source_file_path.split("/")[-1].split(".")[-2]
        + "_mfcc.csv",
        "w",
    )
    writer = csv.writer(myFile)
    writer.writerow(np.linspace(1, 13, 13, dtype=int))
    writer.writerows(MFCCs.transpose())


video_dir = "../video/"
audio_dir = "../audio/"
mfcc_dir = "../mfcc/"
processed_dir = "../openface_output/"

if not os.path.exists(audio_dir):
    os.mkdir(audio_dir)
if not os.path.exists(mfcc_dir):
    os.mkdir(mfcc_dir)
if not os.path.exists(processed_dir):
    os.mkdir(processed_dir)

video_name_list = [
    item
    for item in os.listdir(video_dir)
    if (".mp4" in item)
    or (".wmv" in item)
    or (".avi" in item)
    or (".webm" in item)
]

# dari video ambil audio taro ke folder audio
for item in video_name_list:
    wav_extract(video_dir + item, audio_dir)
audio_name_list = [item for item in os.listdir(audio_dir) if (".wav" in item)]

# dari audio ambil mfcc taro di folder mfcc
for item in audio_name_list:
    mfcc_extract(audio_dir + item, mfcc_dir)

# dari video ambil facs taro di folder facs
for item in video_name_list:
    subprocess.run(
        [
            "/home/alkhemi/OpenFace/build/bin/FeatureExtraction",
            "-3Dfp",
            "-pose",
            # "-tracked",
            "-f",
            video_dir + item,
            "-out_dir",
            processed_dir,
        ]
    )
