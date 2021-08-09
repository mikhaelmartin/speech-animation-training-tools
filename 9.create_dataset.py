import gc
import os
import numpy as np
import pandas as pd
import time
import multiprocessing

features_dir = "../features_60fps/"
output_data_dir = "../npy_data_mfcc39_landmark107"

mfcc_labels = (
    ["mfcc_" + str(i) for i in range(14)[1:]]
    + ["delta_mfcc_" + str(i) for i in range(14)[1:]]
    + ["delta2_mfcc_" + str(i) for i in range(14)[1:]]
)

# all
landmark_indexes = (
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    + [17, 18, 19, 20, 21]
    + [27, 28, 29, 30, 31, 32, 33]
    + [36, 37, 38, 39, 40, 41]
    + [48, 49, 50, 51]
    + [57, 58, 59, 60, 61, 62]
    + [66, 67]
)
center_indexes = [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]

# exclude X of center landmarks
landmark_labels = ["X_" + str(i) for i in landmark_indexes if i not in center_indexes]
landmark_labels += ["Y_" + str(i) for i in landmark_indexes]
landmark_labels += ["Z_" + str(i) for i in landmark_indexes]

if not os.path.exists(output_data_dir):
    os.mkdir(output_data_dir)

speaker_list = [
    item for item in os.listdir(features_dir) if os.path.isdir(features_dir + item)
]
speaker_list.sort()


def create_data_list(speaker_dir, speaker, filename):
    print("processing", speaker, filename)
    dis_landmark_df = pd.read_csv(
        features_dir + speaker + "/displacement_landmarks/" + filename + "_dis.csv",
        dtype="float16",
    )
    mfcc_df = pd.read_csv(
        features_dir + speaker + "/mfcc/" + filename + "_mfcc.csv", dtype="float16"
    )

    # memastikan jumlah frame x dan y sama dengan cara merge ke dalam satu database
    merged_df = pd.merge(
        dis_landmark_df.loc[:, landmark_labels],
        mfcc_df.loc[:, mfcc_labels],
        left_index=True,
        right_index=True,
        how="inner",
    )

    frame_count = len(merged_df)

    x = [None] * frame_count
    y = [None] * frame_count
    for i in range(frame_count):
        x[i] = merged_df.loc[i, mfcc_labels]
        y[i] = merged_df.loc[i, landmark_labels]
    np.save(speaker_dir + "/" + filename + "_X.npy", np.array(x))
    np.save(speaker_dir + "/" + filename + "_Y.npy", np.array(y))
    return


if __name__ == "__main__":
    start_time = time.time()

    for speaker in speaker_list:
        print("processing", speaker)

        output_speaker_dir = output_data_dir + "/" + speaker

        if not os.path.exists(output_speaker_dir):
            os.mkdir(output_speaker_dir)

        filenames = [
            f.split("_")[0]
            for f in os.listdir(features_dir + speaker + "/displacement_landmarks")
            if f[-4:] == ".csv"
        ]

        pool = multiprocessing.Pool(10)
        pool.starmap(
            create_data_list,
            zip(
                [output_speaker_dir] * len(filenames),
                [speaker] * len(filenames),
                filenames,
            ),
        )
        pool.close()
        pool.join()

        print(gc.get_count())
        print(gc.collect())
        print(gc.get_count())

    print("finished in", time.time() - start_time, "seconds")
exit()
