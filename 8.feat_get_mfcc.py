import os
import numpy as np
import multiprocessing
import librosa
import csv
import time

frame_length = 0.02  # 20ms
audio_sample_rate = 44100
hop_length = int(audio_sample_rate*frame_length)  # 882 frames for 20 ms

audio_dir = "../audio_filtered/"
features_dir = "../features/"
if not os.path.exists(features_dir):
    os.mkdir(features_dir)


def librosa_mfcc_extract(
    speaker,
    audio_file_name
):
    signal, sr = librosa.load(audio_dir + speaker +
                              "/" + audio_file_name, sr=audio_sample_rate)
    MFCCs = librosa.feature.mfcc(
        signal,
        sr=sr,
        n_fft=512,  # window size
        hop_length=hop_length,
        n_mfcc=13,  # num of coefficients
        n_mels=40,  # num of mel band filters
        window="hann",
        htk=True,
    )
    MFCCs = MFCCs.transpose()
    myFile = open(
        features_dir
        + speaker + "/mfcc/" + audio_file_name.split(".")[-2]
        + "_mfcc.csv",
        "w",
    )
    writer = csv.writer(myFile)
    writer.writerow(["frame", "face_id", "timestamp"] +
                    ["mfcc_" + str(i + 1) for i in range(13)])
    face_id = int(speaker.split('s')[-1])
    for i, MFCC in enumerate(MFCCs):
        writer.writerow([i+1] + [face_id] +
                        [np.round(i*frame_length, 3)] +
                        [np.round(x, 3) for x in MFCC.tolist()])
    myFile.close()


speaker_list = [item for item in os.listdir(
    audio_dir) if os.path.isdir(audio_dir + item)]
speaker_list.sort()

if __name__ == "__main__":
    start_time = time.time()
    print("Getting MFCCs")
    for speaker in speaker_list:
        print("Processing " + speaker)
        if not os.path.exists(features_dir + speaker):
            os.mkdir(features_dir+speaker)
        if not os.path.exists(features_dir + speaker + "/mfcc"):
            os.mkdir(features_dir + speaker + "/mfcc")

        audio_name_list = [item for item in os.listdir(
            audio_dir+speaker) if item[-4:] == ".wav"]
        pool = multiprocessing.Pool(10)

        # dari audio ambil mfcc taro di folder mfcc
        pool.starmap(
            librosa_mfcc_extract,
            zip([speaker]*len(audio_name_list), audio_name_list)
        )
        pool.close()
        pool.join()
    print("Finished in", time.time()-start_time, "seconds")
    exit()
