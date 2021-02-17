import os
import numpy as np
import pandas as pd
from tensorflow.keras.metrics import (
    MeanSquaredError,
    RootMeanSquaredError,
    CosineSimilarity,
)


true_dir = "../features_60fps/s19/landmarks/"
# pred_dir = "../predictions/RNN64_s19_batch64_seq0-0-0-0-1675-0.0112.h5/s19/test/"
# pred_dir = "../predictions/GRU64_s19_batch64_seq0-0-0-0-320-0.0104.h5/s19/test/"
pred_dir = "../predictions/LSTM64_s19_batch64_seq0-0-0-0-479-0.0102.h5/s19/test/"
# pred_dir = "../predictions/Attention64_s19_mha1_numheads8_ffn0_batch64_seq12-12-0-0-21-0.0102.h5/s19/test/"
# pred_dir = "../predictions/Attention64_s19_mha1_numheads8_ffn1_dff256_batch64_seq12-12-0-0-07-0.0099.h5/s19/test/"


num_past_x_frame = 0
num_future_x_frame = 0
num_past_y_frame = 0
num_future_y_frame = 0
max_past = max(num_past_x_frame, num_past_y_frame)
max_future = max(num_future_x_frame, num_future_y_frame)

evaluation_files = [f[:-9] for f in os.listdir(pred_dir) if f[-9:] == "_pred.csv"]

mse = MeanSquaredError()
rmse = RootMeanSquaredError()
cs = CosineSimilarity(axis=0)


true_values = []
pred_values = []
for file in evaluation_files:
    true_df = pd.read_csv(true_dir + file + ".csv")
    pred_df = pd.read_csv(pred_dir + file + "_pred.csv")
    length = min(len(true_df), len(pred_df))

    true_values.append(
        list(
            true_df.loc[max_past : max_past + length - 1, "X_0":"Z_67"]
            .to_numpy()
            .flatten()
        )
    )

    pred_values.append(
        list(pred_df.loc[0 : length - 1, "X_0":"Z_67"].to_numpy().flatten())
    )


mse.update_state(true_values, pred_values)
rmse.update_state(true_values, pred_values)
cs.update_state(true_values, pred_values)
r = np.corrcoef(np.array(true_values).flatten(), np.array(pred_values).flatten())
print(r)
print(len(true_values))
print(mse.result().numpy(), rmse.result().numpy(), cs.result().numpy())
