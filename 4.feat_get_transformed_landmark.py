# tools for reorienting the face to look at camera
# and interpolate data with specific frame length
# TODO interpolasinya masih salah

import os
import multiprocessing
import scipy.interpolate
import pandas as pd
import vpython as vp
import numpy as np
import time
from math import floor
# target frame length
target_frame_length = 1 / 60  # 0.02  # 20ms

openface_dir = "../openface/"
features_dir = "../features/"
if not os.path.exists(features_dir):
    os.mkdir(features_dir)


def get_tranformed_data(row):
    for i in range(68):
        # set origin to (0,0,0)
        x = float(row["X_" + str(i)]) - float(row["pose_Tx"])
        y = float(row["Y_" + str(i)]) - float(row["pose_Ty"])
        z = float(row["Z_" + str(i)]) - float(row["pose_Tz"])

        v = vp.vector(x, y, z)
        # set orientation facing the camera
        v = v.rotate(float(row["pose_Rx"]), vp.vector(-1, 0, 0))
        v = v.rotate(float(row["pose_Ry"]), vp.vector(0, -1, 0))
        v = v.rotate(float(row["pose_Rz"]), vp.vector(0, 0, -1))
        # somehow the output is inverted in y axis
        # so negative value is needed
        row["X_" + str(i)] = v.x
        row["Y_" + str(i)] = -v.y
        row["Z_" + str(i)] = v.z
    return row


def process_landmarks(speaker, filename):
    print("processing", speaker, filename)
    df = pd.read_csv(openface_dir + speaker + "/" + filename, skipinitialspace=True)
    # transform data so the face facing the camera
    for row_index, row in df.iterrows():
        df.loc[row_index] = get_tranformed_data(row)

    functions = {}
    functions["confidence"] = scipy.interpolate.interp1d(
        [float(item) for item in df["timestamp"]],
        [float(item) for item in df["confidence"]],
        kind="linear",
    )
    functions["success"] = scipy.interpolate.interp1d(
        [float(item) for item in df["timestamp"]],
        [int(item) for item in df["success"]],
        kind="previous",
    )
    axis_names = []
    for axis in ["X_", "Y_", "Z_"]:
        for i in range(68):
            axis_names.append(axis + str(i))
            functions[axis + str(i)] = scipy.interpolate.interp1d(
                [float(item) for item in df["timestamp"]],
                [float(item) for item in df[axis + str(i)]],
                kind="cubic",
            )
    face_id = int(df.loc[0, "face_id"])

    timestamp_bound = df.loc[len(df) - 1, "timestamp"]
    length = round(timestamp_bound / target_frame_length)
    new_df = pd.DataFrame()
    new_df["frame"] = [(i + 1) for i in range(length)]
    new_df["face_id"] = [(face_id) for i in range(length)]
    new_df["timestamp"] = [round(i * target_frame_length, 4) for i in range(length)]
    new_df["confidence"] = [(functions["confidence"](x)) for x in new_df["timestamp"]]
    new_df["success"] = [int(functions["success"](x)) for x in new_df["timestamp"]]
    for axis in ["X_", "Y_", "Z_"]:
        for j in range(68):
            new_df[axis + str(j)] = [
                np.round(functions[axis + str(j)](x), 3) for x in new_df["timestamp"]
            ]
    new_df.to_csv(features_dir + speaker + "/landmarks/" + filename, index=False)


speaker_list = [
    item for item in os.listdir(openface_dir) if os.path.isdir(openface_dir + item)
]
speaker_list.sort()
print(speaker_list)

if __name__ == "__main__":
    start_time = time.time()
    print("Getting landmarks")
    for speaker in speaker_list:
        # print("Processing " + speaker)
        if not os.path.exists(features_dir + speaker):
            os.mkdir(features_dir + speaker)
        if not os.path.exists(features_dir + speaker + "/landmarks"):
            os.mkdir(features_dir + speaker + "/landmarks")

        raw_openface_list = [
            item for item in os.listdir(openface_dir + speaker) if item[-4:] == ".csv"
        ]

        pool = multiprocessing.Pool()

        # dari openface ambil landmark taro di folder landmark
        pool.starmap(
            process_landmarks,
            zip([speaker] * len(raw_openface_list), raw_openface_list),
        )
        pool.close()
        pool.join()
    print("finished in", time.time() - start_time, "seconds")
    exit()
