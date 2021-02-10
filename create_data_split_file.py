import os
import numpy as np
import pandas as pd
import time
import multiprocessing

# sequence to sequence parameter
num_past_x_frame = 9
num_future_x_frame = 0
num_past_y_frame = 0
num_future_y_frame = 0

data_ratio = 0.9
features_dir = "../features_60fps/"
output_data_dir = (
    "../npy_data_s19_mfcc39_landmark107"
    + "_pastX"
    + str(num_past_x_frame)
    + "_futureX"
    + str(num_future_x_frame)
    + "_pastY"
    + str(num_past_y_frame)
    + "_futureY"
    + str(num_future_y_frame)
    + "/"
)

mfcc_labels = (
    ["mfcc_" + str(i) for i in range(14)[1:]]
    + ["delta_mfcc_" + str(i) for i in range(14)[1:]]
    + ["delta2_mfcc_" + str(i) for i in range(14)[1:]]
)

# only around mouth
# data_dir = "../npy_data_mouth/"
# landmark_indexes = [5, 6, 7, 8, 31, 32, 33, 48,
#                     49, 50, 51, 57, 58, 59, 60, 61, 62, 66, 67]
# center_indexes = [8, 33, 51, 57, 62, 66]

# all except eye
# data_dir = "../npy_data_no_eye/"
# landmark_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 27, 28,
#                     29, 30, 31, 32, 33, 48, 49, 50, 51, 57, 58, 59, 60, 61,
#                     62, 66, 67]
# center_indexes = [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]

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


def create_data_list(speaker, filename):
    print("processing", speaker, filename)
    dis_landmark_df = pd.read_csv(
        features_dir + speaker + "/displacement_landmarks/" + filename + "_dis.csv",
        dtype="float16",
    )
    mfcc_df = pd.read_csv(
        features_dir + speaker + "/mfcc/" + filename + "_mfcc.csv", dtype="float16"
    )
    merged_df = pd.merge(
        dis_landmark_df.loc[:, landmark_labels],
        mfcc_df.loc[:, mfcc_labels],
        left_index=True,
        right_index=True,
        how="inner",
    )

    data_size = len(merged_df) - (
        max(num_past_x_frame, num_past_y_frame)
        + max(num_future_x_frame, num_future_y_frame)
    )
    x = [None] * data_size
    y = [None] * data_size
    for i in range(data_size):
        current_frame_index = i + max(num_past_x_frame, num_past_y_frame)
        x[i] = merged_df.loc[
            current_frame_index
            - num_past_x_frame : current_frame_index
            + num_future_x_frame,
            mfcc_labels,
        ]
        y[i] = merged_df.loc[
            current_frame_index
            - num_past_y_frame : current_frame_index
            + num_future_y_frame,
            landmark_labels,
        ]
    return x, y


if __name__ == "__main__":
    start_time = time.time()

    X_train = list()
    Y_train = list()

    X_test = list()
    Y_test = list()

    train_list = list()
    test_list = list()

    for speaker in speaker_list[18:19]:
        print("processing", speaker)

        filenames = [
            f.split("_")[0]
            for f in os.listdir(features_dir + speaker + "/displacement_landmarks")
            if f[-4:] == ".csv"
        ]

        split_index = int(round(len(filenames) * data_ratio))
        filenames_train = filenames[:split_index]
        filenames_test = filenames[split_index:]
        pd.DataFrame(
            zip(
                filenames_train,
                filenames_test + [""] * (len(filenames_train) - len(filenames_test)),
            ),
            columns=["train", "test"],
        ).to_csv(output_data_dir + speaker + "_split.csv", index=False)
        train_list.extend(filenames_train)
        test_list.extend(filenames_test)

        # train data
        pool = multiprocessing.Pool(10)
        data = pool.starmap(
            create_data_list, zip([speaker] * len(filenames_train), filenames_train)
        )
        pool.close()
        pool.join()

        for x, y in data:
            X_train.extend(x)
            Y_train.extend(y)

        # test data
        pool = multiprocessing.Pool(10)
        data = pool.starmap(
            create_data_list, zip([speaker] * len(filenames_test), filenames_test)
        )
        pool.close()
        pool.join()

        for x, y in data:
            X_test.extend(x)
            Y_test.extend(y)

    print(
        "\nX_train\t",
        np.array(X_train, dtype=float).shape,
        "\nY_train\t",
        np.array(Y_train, dtype=float).shape,
        "\nX_test\t",
        np.array(X_test, dtype=float).shape,
        "\nY_test\t",
        np.array(Y_test, dtype=float).shape,
    )
    np.save(output_data_dir + "X_train.npy", np.array(X_train, dtype=float))
    np.save(output_data_dir + "Y_train.npy", np.array(Y_train, dtype=float))
    np.save(output_data_dir + "X_test.npy", np.array(X_test, dtype=float))
    np.save(output_data_dir + "Y_test.npy", np.array(Y_test, dtype=float))

    pd.DataFrame(
        zip(train_list, test_list + [""] * (len(train_list) - len(test_list))),
        columns=["train", "test"],
    ).to_csv(output_data_dir + "split.csv", index=False)
    print("finished in", time.time() - start_time, "seconds")
exit()
