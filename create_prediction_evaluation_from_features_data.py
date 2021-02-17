import os
import numpy as np
import pandas as pd
import time
from attention import (
    PositionalEncoding,
    MultiHeadAttentionSubLayer,
    FeedForwardNetworkSubLayer,
)
import multiprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, GRU, Flatten


start_time = time.time()
# prediksi dilakukan hanya untuk yang ada split filenya aja
data_dir = "../npy_data_s19_mfcc39_landmark107_pastX12_futureX12_pastY0_futureY0/"
speakers = [
    file.split("_")[0] for file in os.listdir(data_dir) if file[-10:] == "_split.csv"
]

features_data = "../features_60fps/"


# load model
# model_name = "RNN64_s19_batch64_seq0-0-0-0"
# model_weights = "RNN64_s19_batch64_seq0-0-0-0-1675-0.0112.h5"
# model_name = "GRU64_s19_batch64_seq0-0-0-0"
# model_weights = "GRU64_s19_batch64_seq0-0-0-0-320-0.0104.h5"
# model_name = "LSTM64_s19_batch64_seq0-0-0-0"
# model_weights = "LSTM64_s19_batch64_seq0-0-0-0-479-0.0102.h5"
# model_name = "Attention64_s19_mha1_numheads8_ffn0_batch64_seq12-12-0-0"
# model_weights = "Attention64_s19_mha1_numheads8_ffn0_batch64_seq12-12-0-0-21-0.0102.h5"
model_name = "Attention64_s19_mha1_numheads8_ffn1_dff256_batch64_seq12-12-0-0"
model_weights = "Attention64_s19_mha1_numheads8_ffn1_dff256_batch64_seq12-12-0-0-07-0.0099.h5"

units = 64
num_past_x_frame = 12
num_future_x_frame = 12
num_past_y_frame = 0
num_future_y_frame = 0
num_heads = 8
dff = 256

input_seq_length = 1 + num_past_x_frame + num_future_x_frame
output_seq_length = 1 + num_past_y_frame + num_future_y_frame
# RNN model
# selalu error kalau pake gpu. memorynya ga cukup
def predict(x):
    with tf.device("/cpu:0"):
        model = Sequential(
            name=model_name,
            layers=[
                Input((input_seq_length, 39)),
                # SimpleRNN(units=units, return_sequences=True, activation="tanh"),
                # GRU(units=units, return_sequences=True, activation="tanh"),
                # LSTM(units=units, return_sequences=True, activation="tanh"),
                Dense(units=units, activation="tanh"),
                PositionalEncoding(position=input_seq_length, d_model=units),
                MultiHeadAttentionSubLayer(d_model=units, num_heads=num_heads),
                FeedForwardNetworkSubLayer(d_model=units, dff=dff),
                tf.keras.layers.Reshape((output_seq_length, units * input_seq_length)),
                Dense(107, activation="linear"),
            ],
        )
        model.load_weights("../saved_model/" + model_name + "/" + model_weights)
        return model.predict(x)


# model = Sequential(
#     name=model_name,
#     layers=[
#         Input((1, 39)),
#         SimpleRNN(units=units, return_sequences=True, activation="tanh"),
#         GRU(units=units, return_sequences=True, activation="tanh"),
#         LSTM(units=units, return_sequences=True, activation="tanh"),
#         Dense(107, activation="linear"),
#     ],
# )

predictions_dir = "../predictions/" + model_weights
if not os.path.isdir("../predictions/"):
    os.mkdir("../predictions/")
if not os.path.isdir(predictions_dir):
    os.mkdir(predictions_dir)


# labels
mfcc_labels = (
    ["mfcc_" + str(i) for i in range(14)[1:]]
    + ["delta_mfcc_" + str(i) for i in range(14)[1:]]
    + ["delta2_mfcc_" + str(i) for i in range(14)[1:]]
)

landmark_indexes = (
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    + [17, 18, 19, 20, 21]
    + [27, 28, 29, 30, 31, 32, 33]
    + [36, 37, 38, 39, 40, 41]
    + [48, 49, 50, 51]
    + [57, 58, 59, 60, 61, 62]
    + [66, 67]
)

mirror_pair_indexes = (
    [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9]]
    + [[17, 26], [18, 25], [19, 24], [20, 23], [21, 22]]
    + [[36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46]]
    + [[31, 35], [32, 34]]
    + [[48, 54], [49, 53], [50, 52], [58, 56], [59, 55]]
    + [[60, 64], [61, 63], [67, 65]]
)

center_indexes = [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]
landmark_labels = ["X_" + str(i) for i in landmark_indexes if i not in center_indexes]
landmark_labels += ["Y_" + str(i) for i in landmark_indexes]
landmark_labels += ["Z_" + str(i) for i in landmark_indexes]

all_landmark_labels = []
for axis in ["X_", "Y_", "Z_"]:
    for i in range(68):
        all_landmark_labels.append(axis + str(i))


def create_predict_csv(speaker, folder, file):
    print("processing ", file)
    X_df = pd.read_csv(
        features_data + speaker + "/mfcc/" + file + "_mfcc.csv",
        usecols=mfcc_labels,
        skipinitialspace=True,
        dtype="float16",
    )

    data_size = len(X_df) - (
        max(num_past_x_frame, num_past_y_frame)
        + max(num_future_x_frame, num_future_y_frame)
    )
    X = np.zeros(shape=(data_size, 1 + num_past_x_frame + num_past_x_frame, 39))

    for i in range(data_size):
        current_frame_index = i + max(num_past_x_frame, num_past_y_frame)
        X[i] = X_df.loc[
            current_frame_index
            - num_past_x_frame : current_frame_index
            + num_future_x_frame,
            mfcc_labels,
        ].to_numpy()

    def custom_scale(arr, a, b):
        return arr * a + b

    def custom_inverse_scale(arr, a, b):
        return (arr - b) / a

    # scale data
    x_scale_coefs = pd.read_csv(data_dir + "x_scale_coef.csv").to_numpy(dtype="float16")
    X_scaled = np.zeros(shape=X.shape, dtype="float16")
    for i in range(X.shape[2]):
        X_scaled[:, :, i] = custom_scale(
            X[:, :, i], x_scale_coefs[i, 0], x_scale_coefs[i, 1]
        )

    # prediction
    Y_scaled = predict(X_scaled)

    # inverse scale
    y_scale_coefs = pd.read_csv(data_dir + "y_scale_coef.csv").to_numpy(dtype="float16")
    Y = np.zeros(shape=Y_scaled.shape, dtype="float16")
    for i in range(Y.shape[2]):
        Y[:, :, i] = custom_inverse_scale(
            Y_scaled[:, :, i], y_scale_coefs[i, 0], y_scale_coefs[i, 1]
        )

    # create all labels
    displacement_landmark_df = pd.DataFrame()
    for key in all_landmark_labels:
        displacement_landmark_df[key] = [0] * len(Y)

    # left side landmarks
    for i in range(len(landmark_labels)):
        displacement_landmark_df[landmark_labels[i]] = Y[:, :, i]

    # assign mirror data
    for left, right in mirror_pair_indexes:
        displacement_landmark_df["X_" + str(right)] = -displacement_landmark_df[
            "X_" + str(left)
        ]
        displacement_landmark_df["Y_" + str(right)] = displacement_landmark_df[
            "Y_" + str(left)
        ]
        displacement_landmark_df["Z_" + str(right)] = displacement_landmark_df[
            "Z_" + str(left)
        ]

    # make sure x value of center landmark is zero displacement
    for i in center_indexes:
        displacement_landmark_df["X_" + str(i)] = [0] * len(displacement_landmark_df)

    # add identity
    identity_landmark_df = pd.read_csv(
        features_data + speaker + "/silent_landmarks/" + file + "_sil.csv"
    )

    landmark_df = pd.DataFrame()
    for key in all_landmark_labels:
        landmark_df[key] = (
            displacement_landmark_df[key] + identity_landmark_df.loc[0, key]
        )

    landmark_df.to_csv(
        predictions_dir + "/" + speaker + "/" + folder + "/" + file + "_pred.csv"
    )


if __name__ == "__main__":
    for speaker in speakers:
        print("processing ", speaker)

        split_df = pd.read_csv(data_dir + speaker + "_split.csv", dtype=str)

        train_files = split_df["train"].dropna()
        test_files = split_df["test"].dropna()

        if not os.path.isdir(predictions_dir + "/" + speaker):
            os.mkdir(predictions_dir + "/" + speaker)
        if not os.path.isdir(predictions_dir + "/" + speaker + "/train"):
            os.mkdir(predictions_dir + "/" + speaker + "/train")
        if not os.path.isdir(predictions_dir + "/" + speaker + "/test"):
            os.mkdir(predictions_dir + "/" + speaker + "/test")

        # warning cpu num veru dependent on memory capacity
        pool = multiprocessing.Pool(10)
        pool.starmap(
            create_predict_csv,
            zip(
                [speaker] * len(train_files), ["train"] * len(train_files), train_files
            ),
        )
        pool.close()
        pool.join()

        pool = multiprocessing.Pool(10)
        pool.starmap(
            create_predict_csv,
            zip([speaker] * len(test_files), ["test"] * len(test_files), test_files),
        )
        pool.close()
        pool.join()

        # for file in train_files:
        #     create_predict_csv(speaker, "train", file)

        # for file in test_files:
        #     create_predict_csv(speaker, "test", file)

    print(time.time() - start_time)
