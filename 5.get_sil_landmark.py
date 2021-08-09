import os
import multiprocessing
import pandas as pd
from numpy import round
from statistics import mean
import time

source_frame_length = 0.00004  # audio file 25kHz
target_frame_length = 1/60  #0.02

features_dir = "../features/"

if not os.path.exists(features_dir):
    print(features_dir, "not found. exiting..")
    exit()

speaker_list = [item for item in os.listdir(
    features_dir) if os.path.isdir(features_dir + item)]
speaker_list.sort()


def get_silent_landmarks(speaker, landmark, align):
    landmark_df = pd.read_csv(features_dir+speaker+"/landmarks/"+landmark)
    align_df = pd.read_csv(features_dir+speaker+"/alignment/"+align)
    data = {}
    for axis in ["X_", "Y_", "Z_"]:
        for index in range(68):
            lm = axis+str(index)
            data[lm] = round(
                mean(
                    [val for i, val in enumerate(landmark_df[lm].values)
                     if align_df["word"].values[i] == "sil"]),
                decimals=3)
    # save file
    df = pd.DataFrame(data, [0])
    df.to_csv(features_dir + speaker +
              "/silent_landmarks/" + landmark.split('.')[0] +
              "_sil.csv", index=False)


if __name__ == "__main__":
    start_time = time.time()
    print("Getting silent landmark")
    for speaker in speaker_list:
        print("Processing " + speaker)
        if not os.path.exists(features_dir + speaker + "/silent_landmarks"):
            os.mkdir(features_dir + speaker + "/silent_landmarks")

        landmarks_file_list = [
            item for item in os.listdir(features_dir+speaker+"/landmarks") if
            item.split('.')[-1] == "csv"]

        align_file_list = [
            item for item in os.listdir(features_dir+speaker+"/alignment") if
            item.split('_')[-1] == "align.csv"]

        if len(landmarks_file_list) != len(align_file_list):
            print("error: landmark and align frames not equal. skipping")
        else:
            pool = multiprocessing.Pool()

            pool.starmap(
                get_silent_landmarks,
                zip(
                    [speaker]*len(landmarks_file_list),
                    landmarks_file_list,
                    align_file_list)
            )
            pool.close()
            pool.join()
    print("Finished in", time.time()-start_time, "seconds")
    exit()
