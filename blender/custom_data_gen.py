from numba.core.errors import new_error_context
import numpy as np
import tensorflow as tf
import os
from math import floor

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
    print(data.shape)
    new_data = np.array( [data[0]]*num_pre)
    print(new_data.shape)
    new_data=np.append(new_data,data,axis=0)
    print(new_data.shape)
    new_data=np.append(new_data,np.array([data[-1]]*num_post),axis=0)
    print(new_data.shape)
    new_data = GetSequenceData(new_data, num_pre, num_post, max_num_pre, max_num_post)
    return np.array(new_data)

class CustomSeqDataGenSeqCached(tf.keras.utils.Sequence):
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

        # dapetin semua file paths
        for root, _, files in os.walk(self.source_dir):
            files.sort()
            for file in files:
                if file.endswith("_X.npy"):
                    self.x_data.append(
                        GetSequenceData(
                            self.x_scaler.scale(np.load(os.path.join(root, file))),
                            self.x_num_pre,
                            self.x_num_post,
                            self.max_num_pre,
                            self.max_num_post,
                        )
                    )
                if file.endswith("_Y.npy"):
                    self.y_data.append(
                        GetSequenceData(
                            self.y_scaler.scale(np.load(os.path.join(root, file))),
                            self.y_num_pre,
                            self.y_num_post,
                            self.max_num_pre,
                            self.max_num_post,
                        )
                    )

        if len(self.x_data) != len(self.y_data):
            print(len(self.x_data), len(self.y_data))
            raise ValueError("X and Y file data count doesn't match")

        self.file_count = len(self.x_data)

        self.x_data_len = 0
        self.y_data_len = 0
        for i in range(self.file_count):
            if len(self.x_data[i]) != len(self.y_data[i]):
                print(len(self.x_data[i]), len(self.y_data[i]))
                raise ValueError("X and Y data length doesn't match")

            self.x_data_len += len(self.x_data[i])
            self.y_data_len += len(self.y_data[i])

        print("datapoints", self.x_data_len)

    def __len__(self):
        return floor(self.x_data_len / self.batch_size)

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        while True:
            if self.file_index >= self.file_count:
                break
            x_batch += self.x_data[self.file_index][self.data_index :]
            y_batch += self.y_data[self.file_index][self.data_index :]

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
        self.file_index = 0
        self.data_index = 0


class CustomSeqDataGenSeqUncached(tf.keras.utils.Sequence):
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

        # dapetin semua file paths
        for root, _, files in os.walk(self.source_dir):
            files.sort()
            for file in files:
                if file.endswith("_X.npy"):
                    self.x_data.append(
                        self.x_scaler.scale(np.load(os.path.join(root, file)))
                    )
                if file.endswith("_Y.npy"):
                    self.y_data.append(
                        self.y_scaler.scale(np.load(os.path.join(root, file)))
                    )

        if len(self.x_data) != len(self.y_data):
            print(len(self.x_data), len(self.y_data))
            raise ValueError("X and Y file data count doesn't match")

        self.file_count = len(self.x_data)

        self.x_data_len = 0
        self.y_data_len = 0
        for i in range(self.file_count):
            if len(self.x_data[i]) != len(self.y_data[i]):
                print(len(self.x_data[i]), len(self.y_data[i]))
                raise ValueError("X and Y data length doesn't match")
            self.x_data_len += self.x_data[i].shape[0] - (
                self.max_num_pre + self.max_num_post
            )
            self.y_data_len += self.y_data[i].shape[0] - (
                self.max_num_pre + self.max_num_post
            )

        print(self.x_data_len)

    def __len__(self):
        return floor(self.x_data_len / self.batch_size)

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        while True:
            if self.file_index >= self.file_count:
                break
            x_batch += GetSequenceData(
                self.x_data[self.file_index].tolist(),
                self.x_num_pre,
                self.x_num_post,
                self.max_num_pre,
                self.max_num_post,
            )[self.data_index :]
            y_batch += GetSequenceData(
                self.y_data[self.file_index].tolist(),
                self.y_num_pre,
                self.y_num_post,
                self.max_num_pre,
                self.max_num_post,
            )[self.data_index :]

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
        self.file_index = 0
        self.data_index = 0


class CustomSeqDataGenFlowFromFile(tf.keras.utils.Sequence):
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

        self.x_filepaths = []
        self.y_filepaths = []
        self.data_index = 0
        self.file_index = 0

        # dapetin semua file paths
        for root, _, files in os.walk(self.source_dir):
            files.sort()
            for file in files:
                if file.endswith("_X.npy"):
                    self.x_filepaths.append(os.path.join(root, file))
                if file.endswith("_Y.npy"):
                    self.y_filepaths.append(os.path.join(root, file))

        self.x_filepaths.sort()
        self.y_filepaths.sort()

        if len(self.x_filepaths) != len(self.y_filepaths):
            print(len(self.x_filepaths), len(self.y_filepaths))
            raise ValueError("X and Y file count doesn't match")

        self.file_count = len(self.x_filepaths)

        # dapetin panjang masing2 file simpen sebagai dictionary butuh juga totalnya total.
        self.x_data_len = 0
        self.y_data_len = 0
        for i in range(self.file_count):
            if len(self.x_filepaths[i]) != len(self.y_filepaths[i]):
                print(len(self.x_filepaths[i]), len(self.y_filepaths[i]))
                raise ValueError("X and Y data length doesn't match")
            self.x_data_len += np.load(self.x_filepaths[i]).shape[0] - (
                self.max_num_pre + self.max_num_post
            )
            self.y_data_len += np.load(self.y_filepaths[i]).shape[0] - (
                self.max_num_pre + self.max_num_post
            )

        print(self.x_data_len)

    def __len__(self):
        return floor(self.x_data_len / self.batch_size)

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        while True:
            if self.file_index >= self.file_count:
                break
            x_batch += GetSequenceData(
                (np.load(self.x_filepaths[self.file_index])).tolist(),
                self.x_num_pre,
                self.x_num_post,
                self.max_num_pre,
                self.max_num_post,
            )[self.data_index :]
            y_batch += GetSequenceData(
                (np.load(self.y_filepaths[self.file_index])).tolist(),
                self.y_num_pre,
                self.y_num_post,
                self.max_num_pre,
                self.max_num_post,
            )[self.data_index :]

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
        return self.x_scaler.scale(np.array(x_batch)), self.y_scaler.scale(
            np.array(y_batch)
        )

    def on_epoch_end(self):
        self.file_index = 0
        self.data_index = 0
