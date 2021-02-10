import os
import multiprocessing
import pandas as pd
from math import ceil
from numpy import round
import time

source_frame_length = 0.00004  # audio file 25kHz
target_frame_length = 1/60    #0.02

alignment_dir = "../alignment/"
features_dir = "../features/"

if not os.path.exists(alignment_dir):
    print(alignment_dir, "not found. exiting..")
    exit()
if not os.path.exists(features_dir):
    os.mkdir(features_dir)

speaker_list = [item for item in os.listdir(
    alignment_dir) if os.path.isdir(alignment_dir + item)]
speaker_list.sort()


def create_align_csv(speaker, align_filename):
    # print("Processing", speaker, align_filename)
    file = open(alignment_dir + speaker + "/" + align_filename, "r")

    align_dict_list = [[data.strip() for data in row.split(" ")]
                       for row in file]
    file.close()

    data_dict = {}

    for frame in range(ceil(float(align_dict_list[-1][1]) *
                            (source_frame_length/target_frame_length))):
        for i in range(len(align_dict_list)):
            if (frame + 1) * target_frame_length/source_frame_length <= \
                    float(align_dict_list[i][1]):
                data_dict[frame] = {
                    "frame": frame+1,
                    "timestamp":
                    round(target_frame_length*frame, decimals=4),
                    "word": align_dict_list[i][2]
                }
                break

    # save file
    df = pd.DataFrame.from_dict(data_dict, "index")
    df.to_csv(features_dir + speaker +
              "/alignment/" + align_filename.split('.')[0] +
              "_align.csv", index=False)


if __name__ == "__main__":
    start_time = time.time()
    print("Getting alignment")
    for speaker in speaker_list:

        print("Processing " + speaker)
        if not os.path.exists(features_dir + speaker):
            os.mkdir(features_dir+speaker)
        if not os.path.exists(features_dir + speaker + "/alignment"):
            os.mkdir(features_dir + speaker + "/alignment")

        align_file_list = [item for item in os.listdir(
            alignment_dir+speaker) if item.split('.')[-1] == "align"]

        pool = multiprocessing.Pool()

        # dari audio ambil mfcc taro di folder mfcc
        pool.starmap(
            create_align_csv,
            zip([speaker]*len(align_file_list), align_file_list)
        )
        pool.close()
        pool.join()
    print(time.time()-start_time)
    exit()
