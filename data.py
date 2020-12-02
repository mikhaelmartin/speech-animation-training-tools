import os
import pandas as pd
import vpython as vp


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


raw_data_dir = "../raw_data/"
output_data_dir = "../data/"
if not os.path.exists(raw_data_dir):
    print("raw data dir does not exists")
if not os.path.exists(output_data_dir):
    os.mkdir(output_data_dir)

# get all raw data names
raw_data_filenames = [
    file for file in os.listdir(raw_data_dir) if file[-4:] == ".csv"
]

for filename in raw_data_filenames:
    print("processing " + filename)
    df = pd.read_csv(raw_data_dir + filename)
    # transform data so the face facing the camera
    for row_index, row in df.iterrows():
        df.loc[row_index] = get_tranformed_data(row)
        if row_index == 0:
            ground = df.loc[0]
        for i in range(68):
            for axis in list(["X_", "Y_", "Z_"]):
                new_value = float(row[axis + str(i)]) - float(
                    ground[axis + str(i)]
                )
                df.loc[row_index, axis + str(i)] = new_value
    df.pop("pose_Tx")
    df.pop("pose_Ty")
    df.pop("pose_Tz")
    df.pop("pose_Rx")
    df.pop("pose_Ry")
    df.pop("pose_Rz")
    df.to_csv(output_data_dir + filename, index=False)
