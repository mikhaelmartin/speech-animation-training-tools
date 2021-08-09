import numpy as np
import pandas as pd


class CustomScaler:
    def __init__(self, feature_size):
        self.feature_size = feature_size
        self.scale_coef = np.zeros(shape=(feature_size, 2))

    def __get_scaler_coef(self, source_min, source_max, target_min, target_max):
        a = (target_max - target_min) / (source_max - source_min)
        b = target_min - (target_max - target_min) * source_min / (
            source_max - source_min
        )
        return a, b

    def scale(self, arr):
        if arr.shape[-1] != self.feature_size:
            raise ValueError("array shape doesn't match")
        shape = arr.shape
        arr = arr.flatten()
        for i in range(self.feature_size):
            arr[i :: self.feature_size] = (
                arr[i :: self.feature_size] * self.scale_coef[i, 0]
                + self.scale_coef[i, 1]
            )
        return arr.reshape(shape)

    def inverse_scale(self, arr):
        if arr.shape[-1] != self.feature_size:
            raise ValueError("array shape doesn't match")
        shape = arr.shape
        arr = arr.flatten()
        for i in range(self.feature_size):
            arr[i :: self.feature_size] = (
                arr[i :: self.feature_size] - self.scale_coef[i, 1]
            ) / self.scale_coef[i, 0]
        return arr.reshape(shape)

    def fit_from_min_max_list(self, data_min_list, data_max_list, target_min, target_max):
        if data_min_list.shape[-1] != self.feature_size or data_max_list.shape[-1] != self.feature_size:
            raise ValueError("array shape doesn't match")
        if target_min >= target_max:
            raise ValueError("target min is greater or equal target max")

        for i in range(self.feature_size):
            self.scale_coef[i] = self.__get_scaler_coef(
                data_min_list[i],
                data_max_list[i],
                target_min,
                target_max,
            )

    def fit_numpy_array(self, arr, target_min, target_max):
        if arr.shape[-1] != self.feature_size:
            raise ValueError("array shape doesn't match")
        if target_min >= target_max:
            raise ValueError("target min is greater or equal target max")

        for i in range(self.feature_size):
            self.scale_coef[i] = self.__get_scaler_coef(
                arr.flatten()[i::self.feature_size].min(),
                arr.flatten()[i::self.feature_size].max(),
                target_min,
                target_max,
            )

    def to_csv(self, file_path):
        pd.DataFrame(data=self.scale_coef, columns=["a", "b"]).to_csv(
            file_path, index=False
        )

    def from_csv(self, file_path):
        data = pd.read_csv(file_path)
        self.scale_coef[:,0] = data["a"]
        self.scale_coef[:,1] = data["b"]
