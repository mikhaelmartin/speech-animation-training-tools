import os
import multiprocessing
import pandas as pd
from numpy import round
import time

features_dir = "../features/"

if not os.path.exists(features_dir):
    print(features_dir, "not found. exiting..")
    exit()

speaker_list = [item for item in os.listdir(
    features_dir) if os.path.isdir(features_dir + item)]
speaker_list.sort()


def get_displacement_landmarks(speaker, landmark, sil_landmark):

    file = open(features_dir+speaker+"/silent_landmarks/"+sil_landmark, "r")
    keys = next(file).split(',')
    values = next(file).split(',')
    sil_landmark = {}
    for key, value in zip(keys, values):
        sil_landmark[key.strip()] = float(value)
    file.close()

    landmark_df = pd.read_csv(features_dir+speaker+"/landmarks/"+landmark)

    for axis in ["X_", "Y_", "Z_"]:
        for index in range(68):
            lm = axis+str(index)
            landmark_df.loc[:, lm] = \
                round(landmark_df.loc[:, lm] -
                      sil_landmark[lm], decimals=3)

    # # save file
    landmark_df.to_csv(features_dir + speaker +
                       "/displacement_landmarks/" + landmark.split('.')[0] +
                       "_dis.csv", index=False)


if __name__ == "__main__":
    start_time = time.time()
    print("Getting displacement landmark")
    for speaker in speaker_list:
        print("Processing " + speaker)
        if not os.path.exists(
                features_dir + speaker + "/displacement_landmarks"):
            os.mkdir(features_dir + speaker + "/displacement_landmarks")

        landmarks_file_list = [
            item for item in os.listdir(features_dir+speaker+"/landmarks") if
            item.split('.')[-1] == "csv"]
        landmarks_file_list.sort()

        sil_landmarks_file_list = [
            item.split('.')[0] + "_sil.csv" for item in landmarks_file_list]
        sil_landmarks_file_list.sort()

        pool = multiprocessing.Pool()

        pool.starmap(
            get_displacement_landmarks,
            zip(
                [speaker]*len(landmarks_file_list),
                landmarks_file_list,
                sil_landmarks_file_list)
        )
        pool.close()
        pool.join()
    print("Finished in", time.time()-start_time, "seconds")
    exit()
