# import bpy
# import sys
# import os
# dir = os.path.dirname(bpy.data.filepath)
# if not dir in sys.path:
#     sys.path.append(dir )

from attention import (
    PositionalEncoding,
    MultiHeadAttentionSubLayer,
    FeedForwardNetworkSubLayer,
)
import tensorflow as tf
from tensorflow.keras.models import Sequential

# from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, GRU, Flatten
from custom_scaler import CustomScaler
from custom_data_gen import GetSequenceData, GetSequenceDataPadded
from mfcc_tools import GetMFCCsFromAudioFile
from landmark_tools import GetDisplacementLandmarkDF


class DisplacementLandmarkPredictor:
    def __init__(
        self,
        X_feature_size=39,
        Y_feature_size=107,
        x_num_pre=12,
        x_num_post=12,
        y_num_pre=0,
        y_num_post=0,
        d_model=64,
        num_heads=8,
        dff=256,
        model_weights="./Attention64_h8_dff256_batch64_lr1e-06-100-0.0036.h5",
        x_scaler="./x_scaler_coef.csv",
        y_scaler="./y_scaler_coef.csv",
    ) -> None:
        self.x_num_pre = x_num_pre
        self.x_num_post = x_num_post
        self.y_num_pre = y_num_pre
        self.y_num_post = y_num_post
        x_seq_len = 1 + x_num_pre + x_num_post
        y_seq_len = 1 + y_num_pre + y_num_post
        self.max_num_pre = max(x_num_pre, y_num_pre)
        self.max_num_post = max(x_num_post, y_num_post)
        self.X_scaler = CustomScaler(X_feature_size)
        self.Y_scaler = CustomScaler(Y_feature_size)
        self.X_scaler.from_csv(x_scaler)
        self.Y_scaler.from_csv(y_scaler)

        model_name = f"Attention{d_model}_h{num_heads}_dff{dff}"
        self.model = Sequential(
            name=model_name,
            layers=[
                # SimpleRNN(units=d_model, return_sequences=True, activation="tanh"),
                # GRU(units=d_model, return_sequences=True, activation="tanh"),
                # LSTM(units=d_model, return_sequences=True, activation="tanh"),
                tf.keras.layers.InputLayer(
                    (x_seq_len, X_feature_size), dtype="float32"
                ),
                tf.keras.layers.Dense(d_model, activation="tanh"),
                PositionalEncoding(position=x_seq_len, d_model=d_model),
                MultiHeadAttentionSubLayer(d_model=d_model, num_heads=num_heads),
                FeedForwardNetworkSubLayer(d_model=d_model, dff=dff),
                tf.keras.layers.Reshape((y_seq_len, d_model * x_seq_len)),
                tf.keras.layers.Dense(Y_feature_size, activation="linear"),
            ],
        )
        self.model.load_weights(model_weights)

    def predict(self, wav_filepath):
        # get X features sequence data: MFCCs, its first and second derivatives
        X = GetSequenceDataPadded(
            GetMFCCsFromAudioFile(wav_filepath),
            self.x_num_pre,
            self.x_num_post,
            self.max_num_pre,
            self.max_num_post,
        )

        # scale input
        X = self.X_scaler.scale(X)

        # predict
        Y = self.model.predict(X)

        # inverse scale output
        return self.Y_scaler.inverse_scale(Y)

    def predictAsDF(self, wav_filepath):
        # get X features sequence data: MFCCs, its first and second derivatives
        X = GetSequenceDataPadded(
            GetMFCCsFromAudioFile(wav_filepath),
            self.x_num_pre,
            self.x_num_post,
            self.max_num_pre,
            self.max_num_post,
        )

        # scale input
        X = self.X_scaler.scale(X)

        # predict
        Y = self.model.predict(X)

        # inverse scale output
        Y = self.Y_scaler.inverse_scale(Y)

        return GetDisplacementLandmarkDF(Y)
