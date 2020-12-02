import os
import pandas as pd

video_dir = "../video/"
features_dir = "../features/"
if not os.path.exists(features_dir):
    print("mfcc folder does not exists")
    exit()
speaker_dir_list = [
    file
    for file in os.listdir(features_dir)
    if os.path.isdir(features_dir + file)
]
speaker_dir_list.sort()

for speaker in speaker_dir_list:
    print("processing " + speaker)
    mfcc_dir = features_dir + speaker + "/mfcc/"
    openface_dir = features_dir + speaker + "/openface/"
    output_dir = "../raw_data/"
    filenames = [
        item[:-4] for item in os.listdir(openface_dir) if item[-4:] == ".csv"
    ]

    for filename in filenames:
        print("processing " + filename)
        df_of = pd.read_csv(openface_dir + filename + ".csv")
        df_mfcc = pd.read_csv(mfcc_dir + filename + "_mfcc.csv")
        for coef in df_mfcc.keys():
            df_of[coef] = df_mfcc[coef]
        df_of.to_csv(output_dir+filename+".csv", index=False)
