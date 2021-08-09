import numpy as np
from numpy.core import numeric
from numpy.core.numeric import zeros_like
import tensorflow as tf
import os
from math import floor, ceil

# fungsi untuk dapat sequence data
def GetSequenceData(data, num_pre, num_post, max_num_pre, max_num_post):
    data_size = len(data) - (max_num_pre + max_num_post)
    data_seq = [None] * data_size
    for i in range(data_size):
        current_frame_index = i + max_num_pre
        data_seq[i] = data[
            current_frame_index - num_pre : current_frame_index + num_post + 1
        ]
    return np.array(data_seq)


# fungsi untuk dapat sequence data
def GetSequenceDataPadded(data, num_pre, num_post, max_num_pre, max_num_post):
    new_data = [data[0]] * num_pre
    new_data.extend(data.tolist())
    new_data.extend([data[-1]] * num_post)
    # datax = GetSequenceData(data, num_pre, num_post, max_num_pre, max_num_post)
    # new_data.extend(GetSequenceData(data, num_pre, num_post, max_num_pre, max_num_post))

    # new_data.extend([data[-1]] * num_post)
    # return np.array(new_data)
    return GetSequenceData(new_data, num_pre, num_post, max_num_pre, max_num_post)


def DataPadding(data, pre_pad_len, post_pad_len, pad_value):
    new_data = [pad_value] * pre_pad_len
    new_data.extend(data.tolist())
    new_data.extend([pad_value] * post_pad_len)
    return np.array(new_data)


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
        remains = arr[-num:]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
        remains = arr[:-num]
    else:
        result[:] = arr
        remains = None
    return result, remains


class CustomDataGenFlowFromFile(tf.keras.utils.Sequence):
    def __init__(
        self,
        source_dir,
        batch_size,
        x_num_pre,
        x_num_post,
        y_num_pre,
        y_num_post,
        x_scaler,
        y_scaler,
    ):
        self.source_dir = source_dir
        self.batch_size = batch_size
        self.x_num_pre = x_num_pre
        self.x_num_post = x_num_post
        self.y_num_pre = y_num_pre
        self.y_num_post = y_num_post
        self.max_num_pre = max(self.x_num_pre, self.y_num_pre)
        self.max_num_post = max(self.x_num_post, self.y_num_post)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        self.x_data = []
        self.y_data = []
        self.data_index = 0
        self.file_index = 0

        # dapetin semua file scaled data
        for root, _, files in os.walk(self.source_dir):
            files.sort()
            for file in files:
                if file.endswith("_X.npy"):
                    self.x_data.append(os.path.join(root, file))
                if file.endswith("_Y.npy"):
                    self.y_data.append(os.path.join(root, file))

        if len(self.x_data) != len(self.y_data):
            print(len(self.x_data), len(self.y_data))
            raise ValueError("X and Y file data count doesn't match")

        self.file_count = len(self.x_data)

        self.x_data_len = 0
        self.y_data_len = 0
        for i in range(self.file_count):
            x = GetSequenceData(
                np.load(self.x_data[i]),
                self.x_num_pre,
                self.x_num_post,
                self.max_num_pre,
                self.max_num_post,
            )
            y = GetSequenceData(
                np.load(self.y_data[i]),
                self.y_num_pre,
                self.y_num_post,
                self.max_num_pre,
                self.max_num_post,
            )
            if len(x) != len(y):
                print(len(x), len(y))
                raise ValueError("X and Y data length doesn't match")

            self.x_data_len += len(x)
            self.y_data_len += len(y)

        print("data files", self.x_data_len)

    def __len__(self):
        # kalau ada sisa dibuang karena yg terakhir ga nyampe batch size
        return floor(self.x_data_len / self.batch_size)

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        while True:
            if self.file_index >= self.file_count:
                break
            x_batch.extend(
                GetSequenceData(
                    self.x_scaler.scale(np.load(self.x_data[self.file_index])),
                    self.x_num_pre,
                    self.x_num_post,
                    self.max_num_pre,
                    self.max_num_post,
                )[self.data_index :]
            )
            y_batch.extend(
                GetSequenceData(
                    self.y_scaler.scale(np.load(self.y_data[self.file_index])),
                    self.y_num_pre,
                    self.y_num_post,
                    self.max_num_pre,
                    self.max_num_post,
                )[self.data_index :]
            )

            if len(x_batch) > self.batch_size:
                self.data_index = self.batch_size - len(x_batch)
                x_batch = x_batch[: self.batch_size]
                y_batch = y_batch[: self.batch_size]
                break
            elif len(x_batch) < self.batch_size:
                self.file_index += 1
                self.data_index = 0
                continue
            else:
                self.file_index += 1
                self.data_index = 0
                break
        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        self.data_index = 0
        self.file_index = 0


class CustomDataGenSeqUnCached(tf.keras.utils.Sequence):
    def __init__(
        self,
        source_dir,
        batch_size,
        x_num_pre,
        x_num_post,
        y_num_pre,
        y_num_post,
        x_scaler,
        y_scaler,
        x_shift=0,
        y_shift=0,
    ):
        self.source_dir = source_dir
        self.batch_size = batch_size
        self.x_num_pre = x_num_pre
        self.x_num_post = x_num_post
        self.y_num_pre = y_num_pre
        self.y_num_post = y_num_post
        self.max_num_pre = max(self.x_num_pre, self.y_num_pre)
        self.max_num_post = max(self.x_num_post, self.y_num_post)
        self.x_seq_len = x_num_pre + x_num_post + 1
        self.y_seq_len = y_num_pre + y_num_post + 1
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.x_file_data = []
        self.y_file_data = []

        # dapetin semua file data
        for root, _, files in os.walk(self.source_dir):
            files.sort()
            for file in files:
                if file.endswith("_X.npy"):
                    self.x_file_data.append(
                        self.x_scaler.scale(np.load(os.path.join(root, file)))
                    )
                if file.endswith("_Y.npy"):
                    self.y_file_data.append(
                        self.y_scaler.scale(np.load(os.path.join(root, file)))
                    )

        if len(self.x_file_data) != len(self.y_file_data):
            print(len(self.x_file_data), len(self.y_file_data))
            raise ValueError("X and Y file count doesn't match")

        # shifting
        if x_shift != 0:
            for file_idx in range(len(self.x_file_data)):
                self.x_file_data[file_idx], _ = shift(
                    self.x_file_data[file_idx], x_shift
                )
        if y_shift != 0:
            for file_idx in range(len(self.y_file_data)):
                self.y_file_data[file_idx], _ = shift(
                    self.y_file_data[file_idx], y_shift
                )
        if x_shift != 0 or y_shift != 0:
            st = max(max(x_shift, 0), max(y_shift, 0))
            fn = min(min(x_shift, 0), min(y_shift, 0))
            if fn == 0:
                fn = None
            for file_idx in range(len(self.x_file_data)):
                self.x_file_data[file_idx] = self.x_file_data[file_idx][st:fn, :]
                self.y_file_data[file_idx] = self.y_file_data[file_idx][st:fn, :]

        # indexing
        self.data_indexes = []  # from data index get file index, seq index
        for file_idx in range(len(self.x_file_data)):
            for seq_idx in range(
                self.max_num_pre, len(self.x_file_data[file_idx]) - self.max_num_post
            ):
                self.data_indexes.append({"file_idx": file_idx, "seq_idx": seq_idx})

        self.data_len = len(self.data_indexes)

        print("datapoints", self.data_len)

    def __len__(self):
        # kalau ada sisa dibuang karena yg terakhir ga nyampe batch size
        return floor(self.data_len / self.batch_size)

    def __getitem__(self, idx):
        index = idx * self.batch_size
        x_batch = [None] * self.batch_size
        y_batch = [None] * self.batch_size
        for i in range(self.batch_size):
            file_idx = self.data_indexes[index + i]["file_idx"]
            seq_idx = self.data_indexes[index + i]["seq_idx"]
            x_batch[i] = self.x_file_data[file_idx][
                seq_idx - self.x_num_pre : seq_idx + self.x_num_post + 1
            ]
            y_batch[i] = self.y_file_data[file_idx][
                seq_idx - self.y_num_pre : seq_idx + self.y_num_post + 1
            ]
        return (np.array(x_batch), np.array(y_batch))

    # def on_epoch_end(self):


class EvaluationDataGen:
    def __init__(
        self,
        source_dir,
        x_scaler,
        y_scaler,
    ):
        self.source_dir = source_dir
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.x_file_data = []
        self.y_file_data = []

        # dapetin semua file data
        for root, _, files in os.walk(self.source_dir):
            files.sort()
            for file in files:
                if file.endswith("_X.npy"):
                    self.x_file_data.append(
                        self.x_scaler.scale(np.load(os.path.join(root, file)))
                    )
                if file.endswith("_Y.npy"):
                    self.y_file_data.append(
                        self.y_scaler.scale(np.load(os.path.join(root, file)))
                    )

        if len(self.x_file_data) != len(self.y_file_data):
            print(len(self.x_file_data), len(self.y_file_data))
            raise ValueError("X and Y file count doesn't match")

        self.data_len = len(self.x_file_data)

        print("datapoints", self.data_len)

    def __len__(self):
        # kalau ada sisa dibuang karena yg terakhir ga nyampe batch size
        return self.data_len

    def getitem_sequence(
        self,
        idx,
        x_num_pre,
        x_num_post,
        y_num_pre,
        y_num_post,
    ):
        max_num_pre = max(x_num_pre, y_num_pre)
        max_num_post = max(x_num_post, y_num_post)
        batch_size = len(self.x_file_data[idx]) - (max_num_pre + max_num_post)
        x_batch = [None] * batch_size
        y_batch = [None] * batch_size
        for batch_idx, seq_idx in enumerate(
            range(max_num_pre, len(self.x_file_data[idx]) - max_num_post)
        ):
            x_batch[batch_idx] = self.x_file_data[idx][
                seq_idx - x_num_pre : seq_idx + x_num_post + 1
            ]
            y_batch[batch_idx] = self.y_file_data[idx][
                seq_idx - y_num_pre : seq_idx + y_num_post + 1
            ]

        return (np.array(x_batch), np.array(y_batch))

    def getitem_single(
        self,
        idx,
        x_shift=0,
        y_shift=0,
        trim_pre=0,
        trim_post=0,
    ):
        batch_size = len(self.x_file_data[idx]) - (trim_pre + trim_post)
        
        x_file_data_shifted, _ = shift(
            self.x_file_data[idx], x_shift, np.zeros_like(self.x_file_data[idx][0])
        )
    
        y_file_data_shifted, _ = shift(
            self.y_file_data[idx], y_shift, np.zeros_like(self.y_file_data[idx][0])
        )
        x_batch = [None] * batch_size
        y_batch = [None] * batch_size
        for batch_idx, seq_idx in enumerate(
            range(trim_pre, len(self.x_file_data[idx]) - trim_post)
        ):
            x_batch[batch_idx] = x_file_data_shifted[seq_idx : seq_idx + 1]
            y_batch[batch_idx] = y_file_data_shifted[seq_idx : seq_idx + 1]
        return (np.array(x_batch), np.array(y_batch))

    # def on_epoch_end(self):


# test
# from custom_scaler import CustomScaler

# data_dir = "/home/alkhemi/Documents/thesis/npy_data_mfcc39_landmark107_s19/test/"
# X_feature_size = 39
# Y_feature_size = 107
# x_num_pre = 11
# x_num_post = 3
# y_num_pre = 11
# y_num_post = 3
# x_pad_value= np.array([-300]+[0.0]*38)
# y_pad_value= np.zeros((107,), dtype=float)
# x_shift = 0
# y_shift = 0
# x_seq_len = x_num_pre + x_num_post + 1
# y_seq_len = y_num_pre + y_num_post + 1
# batch_size = 64
# X_scaler = CustomScaler(X_feature_size)
# Y_scaler = CustomScaler(Y_feature_size)
# X_scaler.from_csv(
#     "/home/alkhemi/Documents/thesis/npy_data_mfcc39_landmark107_s19/train/x_scaler_coef.csv"
# )
# Y_scaler.from_csv(
#     "/home/alkhemi/Documents/thesis/npy_data_mfcc39_landmark107_s19/train/y_scaler_coef.csv"
# )
# data_generator = CustomTransformerDataGenSeqUnCached(
#     data_dir,
#     batch_size,
#     x_num_pre,
#     x_num_post,
#     y_num_pre,
#     y_num_post,
#     X_scaler,
#     Y_scaler,
#     x_pad_value=x_pad_value,
#     y_pad_value=y_pad_value,
# )


# data = data_generator[len(data_generator)-1]
# print( data[0][0].shape, data[0][1].shape, data[1].shape)
