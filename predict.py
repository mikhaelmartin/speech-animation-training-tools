import math
import tensorflow as tf
import os
import numpy as np
from attention import (
    Encoder,
)
from custom_smoothing import Smoothing
from custom_scaler import CustomScaler
from mfcc_tools import GetMFCCsFromVideoFile, GetMFCCsFromAudioFile
from custom_data_gen import DataPadding
from landmark_tools import SaveDisplacementLandmark

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

source_dir = "/mnt/data/Documents/unity-projects/Speech Animation Test/Assets/Resources/Test/"
result_dir = "/mnt/data/Documents/unity-projects/Speech Animation Test/Assets/Resources/Test/"
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

data_dir = "/home/alkhemi/Documents/thesis/npy_data_mfcc39_landmark107/"
single_or_multi = "multi"

train_dir = data_dir + "train/"

X_feature_size = 39
Y_feature_size = 107
x_pad_value = np.array([-300] + [0.0] * 38)
y_pad_value = np.zeros((107,), dtype=float)
X_scaler = CustomScaler(X_feature_size)
Y_scaler = CustomScaler(Y_feature_size)
X_scaler.from_csv(train_dir + "x_scaler_coef.csv")
Y_scaler.from_csv(train_dir + "y_scaler_coef.csv")
model_dir = "/home/alkhemi/Documents/thesis/saved_model/"

with tf.device("/gpu:0"):
    # transformer
    x_num_pre = 11
    x_num_post = 2
    y_num_pre = 11
    y_num_post = 2
    x_seq_len = x_num_pre + x_num_post + 1
    y_seq_len = y_num_pre + y_num_post + 1
    d_model = 64
    num_heads = 8
    dff = 4 * d_model
    num_layers = 1
    model_name = f"TransformerEncoder_d{d_model}_l{num_layers}_h{num_heads}_seq{x_num_pre}-{x_num_post}-{y_num_pre}-{y_num_post}_{single_or_multi}"
    input = tf.keras.Input((x_seq_len, X_feature_size), dtype="float32")
    h = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=x_seq_len,
    )(input)
    output = tf.keras.layers.Dense(Y_feature_size, activation="linear")(h)
    model = tf.keras.models.Model(inputs=input, outputs=output, name=model_name)

    # CNN    
    # x_num_pre = 11
    # x_num_post = 2
    # y_num_pre = 0
    # y_num_post = 0
    # x_seq_len = x_num_pre + x_num_post + 1
    # y_seq_len = y_num_pre + y_num_post + 1
    # filter = 64
    # model_name = f"CNN{filter}_seq{x_num_pre}-{x_num_post}-{y_num_pre}-{y_num_post}_{single_or_multi}"
    # input = tf.keras.Input((x_seq_len,X_feature_size), dtype='float32')
    # x = tf.keras.layers.Reshape((x_seq_len, X_feature_size, 1)) (input)  
    # for i in reversed(range(math.floor(math.log(X_feature_size,2))-1)):
    #     x = tf.keras.layers.Conv2D(filter/(2**i), kernel_size=(1,3), strides =(1,2), activation='relu') (x)
    # seq = x.shape[1]
    # while seq > 1:
    #     if seq > 2:
    #         x = tf.keras.layers.Conv2D(filter, kernel_size=(3,1), strides =(2,1), activation='relu') (x)
    #     else:
    #         x = tf.keras.layers.Conv2D(filter, kernel_size=(2,1), strides =(1,1), activation='relu') (x)
    #     seq = x.shape[1]  
    # x = tf.keras.layers.Reshape((1,filter)) (x)
    # output = tf.keras.layers.Dense(Y_feature_size, activation='linear') (x)
    # model = tf.keras.models.Model(inputs=input,outputs=output, name=model_name)

    save_dir = model_dir+model_name+"/"

    models_weights = [item for item in os.listdir(save_dir) if item.endswith('.h5')]
    models_weights.sort()

    if(len(models_weights)>0):
        model.load_weights(save_dir+models_weights[-1])
    else:
        ValueError("No model weights")


# direct (CNN)
# def predict(X):
#     return np.array(model(X))[:,0,:] 

smoothing = Smoothing(y_seq_len, Y_feature_size, 1, y_num_pre)

def predict(X):
    Y_batch_pred = np.array(model(X))
    Y = np.empty((X.shape[0], Y_feature_size))
    smoothing.reset()
    for i in range(X.shape[0]):
        Y[i] = smoothing(Y_batch_pred[i])
    return Y

# get  all list of filenames
file_list = []
for root, folders, files in os.walk(source_dir):
    for file in files:
        target_dir = result_dir
        for dir in root.split(source_dir)[-1].split("/"):
            target_dir = os.path.join(target_dir, dir)
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)
        if (
            file.endswith(".webm")
            or file.endswith(".mpg")
            or file.endswith(".mp4")
            or file.endswith(".mkv")
            or file.endswith(".mp3")
            or file.endswith(".wav")
        ):
            file_list.append(
                {"source_dir": root, "target_dir": target_dir, "file_name": file}
            )

for count, file in enumerate(file_list):
    print("processing", count + 1, "from", len(file_list))
    # mfcc
    if file["file_name"].endswith(".wav") or file["file_name"].endswith(".mp3"):
        data = GetMFCCsFromAudioFile(os.path.join(file["source_dir"], file["file_name"]))
    else:    
        data = GetMFCCsFromVideoFile(os.path.join(file["source_dir"], file["file_name"]))
    data_len = len(data)
    # pad data
    data = DataPadding(
        data, max(x_num_pre, y_num_pre), max(x_num_post, y_num_post), x_pad_value
    )
    # scale data
    data = X_scaler.scale(data)
    # create seq data
    X = np.empty((data_len, x_seq_len, X_feature_size))
    for i in range(data_len):
        X[i] = data[i : i + x_seq_len]
    # predict
    Y = predict(X)
    # inverse scale
    Y = Y_scaler.inverse_scale(Y)
    # save displacement landmark
    SaveDisplacementLandmark(
        os.path.join(
            file["target_dir"], file["file_name"].split(".")[0] + "_dis_pred.csv"
        ),
        Y,
    )
