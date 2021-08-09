import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name, save_dir="./", last_only=False):
        # self.model = model
        self.save_dir = save_dir
        self.model_name = model_name
        self.last_only = last_only
        self.on_train_begin

    def on_train_begin(self, logs={}):
        if os.path.exists(self.save_dir + self.model_name + "-history.csv"):
            hist_df = pd.read_csv(self.save_dir + self.model_name + "-history.csv")
            self.i = hist_df["epoch"].values.tolist()[-1] + 1
            self.x = hist_df["epoch"].values.tolist()
            self.timestamp = hist_df["timestamp"].values.tolist()
            self.time_offset = hist_df["timestamp"].values.tolist()[-1]

            self.metrics = {}
            for key in hist_df.keys():
                if key != "epoch" and key != "timestamp":
                    self.metrics[key] = hist_df[key].values.tolist()

        else:
            self.i = 1
            self.x = []
            self.timestamp = []
            self.time_offset = 0

            self.metrics = {}

        self.fig = plt.figure()
        self.logs = []
        self.st_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.i += 1
        self.timestamp.append(time.time() - self.st_time + self.time_offset)

        for key in logs.keys():
            if self.metrics.get(key) == None:
                self.metrics[key] = [logs[key]]
            else:
                self.metrics[key].append(logs.get(key))

        # print("saving model")
        # subclassed model can only save weights
        self.model.save_weights(f"{self.save_dir+self.model_name}-{self.x[-1]:04d}.h5")
        # f'-{logs["val_cosine_similarity"]:.4f}.h5')

        if self.last_only:
            if len(self.x) > 1:
                prev_model = next(
                    filter(
                        lambda file: file.endswith(f"{self.x[-2]:04d}.h5"),
                        os.listdir(self.save_dir),
                    ),
                    None,
                )
                if prev_model != None:
                    os.remove(os.path.join(self.save_dir, prev_model))

        # print("saving history")
        hist_df = pd.DataFrame()
        hist_df["epoch"] = self.x
        hist_df["timestamp"] = self.timestamp

        for key in logs.keys():
            hist_df[key] = self.metrics[key]
        hist_df.to_csv(self.save_dir + self.model_name + "-history.csv", index=False)

        clear_output(wait=True)

        # keys = [key for key in self.metrics.keys() if key[0:4] != "val_"]
        # metrics_count = len(keys)
        # f, ax = plt.subplots(ceil(metrics_count/2), 2, sharex=True, figsize=(25,5*ceil(metrics_count/2)))

        # for i, key in enumerate(keys):
        #   r = floor(i/2)
        #   c = i%2
        #   ax[r,c].set_yscale('log')
        #   ax[r,c].set_title(key)
        #   ax[r,c].set_xlabel("epoch")
        #   ax[r,c].plot(self.x, self.metrics[key], label=key)
        #   ax[r,c].plot(self.x, self.metrics["val_"+key], label="val_"+key)
        #   ax[r,c].legend()

        _, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(10, 4))

        ax1.set_yscale("log")
        ax1.set_title("Loss: Mean Squared Error")
        ax1.set_xlabel("epoch")
        ax1.plot(self.x, self.metrics["loss"], label="loss")
        ax1.plot(self.x, self.metrics["val_loss"], label="val_loss")
        ax1.legend()

        # ax2.set_title("Cosine Similarity")
        # ax2.set_xlabel("epoch")
        # ax2.plot(self.x, self.metrics["cosine_similarity"], label="cosine_similarity")
        # ax2.plot(self.x, self.metrics["val_cosine_similarity"], label="val_cosine_similarity")
        # ax2.legend()

        plt.savefig(self.save_dir + "train-mse")
        plt.show()
        print("time elapsed:", time.time() - self.st_time)
