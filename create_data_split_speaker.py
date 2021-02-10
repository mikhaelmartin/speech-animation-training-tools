import os
import numpy as np
import pandas as pd
import time
import multiprocessing

speakers_test = ["s29", "s31", "s32", "s33"]
data_dir = "../npy_data_split_speaker_all/"

seq_length = 10
# data_ratio = 0.9
features_dir = "../features/"

mfcc_labels = ["mfcc_"+str(i) for i in range(14)[1:]]

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
landmark_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 27, 28,
                    29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 48, 49, 50,
                    51, 57, 58, 59, 60, 61, 62, 66, 67]
center_indexes = [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]

# exclude X of center landmarks
landmark_labels = ["X_"+str(i)
                   for i in landmark_indexes if i not in center_indexes]
landmark_labels += ["Y_"+str(i) for i in landmark_indexes]
landmark_labels += ["Z_"+str(i) for i in landmark_indexes]

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

speaker_list = [item for item in os.listdir(features_dir)
                if os.path.isdir(features_dir+item)]
speaker_list.sort()

speakers_train = [s for s in speaker_list if s not in speakers_test]

pd.DataFrame(
    zip(speakers_train, speakers_test+[""]
        * (len(speakers_train)-len(speakers_test))),
    columns=["train", "test"]) \
    .to_csv(data_dir+"split.csv", index=False)


def create_data_list(speaker, filename):
    print("processing", speaker, filename)
    dis_landmark_df = \
        pd.read_csv(features_dir+speaker +
                    "/displacement_landmarks/"+filename+"_dis.csv",
                    dtype='float16')
    mfcc_df = pd.read_csv(features_dir+speaker +
                          "/mfcc/"+filename+"_mfcc.csv",
                          dtype='float16')
    merged_df = pd.merge(dis_landmark_df.loc[:, landmark_labels],
                         mfcc_df.loc[:, mfcc_labels],
                         left_index=True,
                         right_index=True,
                         how="inner")

    x = [merged_df.loc[(i):(i+seq_length-1), mfcc_labels].values
         for i in range(len(merged_df)-(seq_length-1))]
    y = [merged_df.loc[(i+seq_length-1), landmark_labels].values
         for i in range(len(merged_df)-(seq_length-1))]
    return x, y


if __name__ == "__main__":
    start_time = time.time()

    X_train = list()
    Y_train = list()

    X_test = list()
    Y_test = list()

    for speaker in speakers_train:
        print("processing", speaker)

        filenames = [f.split("_")[0] for f in
                     os.listdir(features_dir+speaker+"/displacement_landmarks")
                     if f[-4:] == ".csv"]

        pool = multiprocessing.Pool(10)
        data = pool.starmap(create_data_list, zip(
            [speaker]*len(filenames), filenames))
        pool.close()
        pool.join()

        for x, y in data:
            X_train.extend(x)
            Y_train.extend(y)

    for speaker in speakers_test:
        print("processing", speaker)

        filenames = [f.split("_")[0] for f in
                     os.listdir(features_dir+speaker+"/displacement_landmarks")
                     if f[-4:] == ".csv"]

        pool = multiprocessing.Pool(10)
        data = pool.starmap(create_data_list, zip(
            [speaker]*len(filenames), filenames))
        pool.close()
        pool.join()

        for x, y in data:
            X_test.extend(x)
            Y_test.extend(y)

    print("\nX_train\t", np.array(X_train).shape,
          "\nY_train\t", np.array(Y_train).shape,
          "\nX_test\t", np.array(X_test).shape,
          "\nY_test\t", np.array(Y_test).shape,
          )
    np.save(data_dir+"X_train.npy", np.array(X_train))
    np.save(data_dir+"Y_train.npy", np.array(Y_train))
    np.save(data_dir+"X_test.npy", np.array(X_test))
    np.save(data_dir+"Y_test.npy", np.array(Y_test))

    print("finished in", time.time()-start_time, "seconds")
exit()
