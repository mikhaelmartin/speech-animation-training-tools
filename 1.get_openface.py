import os
import subprocess
import multiprocessing
import pandas as pd
import time

openface_installed = "/home/alkhemi/OpenFace/build/bin/"
video_dir = "../video/"
openface_dir = "../openface/"

if not os.path.exists(openface_dir):
    os.mkdir(openface_dir)

# video is inside a folder for each speaker
# the folder should in format s{id}
# ex: s01
speaker_dir_list = [
    file for file in os.listdir(video_dir) if os.path.isdir(video_dir + file)
]
speaker_dir_list.sort()
print(speaker_dir_list)


def write_face_id(speaker, openface_filename):
    df = pd.read_csv(openface_dir+speaker+"/" +
                     openface_filename, skipinitialspace=True)
    df.loc[:, "face_id"] = int(speaker.split("s")[-1])
    df.loc[:, "pose_Rx"] = df.loc[:, "pose_Rx"].round(3)
    df.loc[:, "pose_Ry"] = df.loc[:, "pose_Ry"].round(3)
    df.loc[:, "pose_Rz"] = df.loc[:, "pose_Rz"].round(3)
    df.to_csv(openface_dir+speaker+"/" + openface_filename,  index=False)


print("getting openface data")
start_time = time.time()
for speaker in speaker_dir_list:
    print("processing " + speaker)
    if not os.path.exists(openface_dir+speaker):
        os.mkdir(openface_dir+speaker)

    video_name_list = [video for video in os.listdir(video_dir+speaker) if
                       (".mp4" in video)
                       or (".wmv" in video)
                       or (".avi" in video)
                       or (".webm" in video)
                       or (".mpg" in video)]

    # get openface raw landmarks output from video
    for video in video_name_list:
        subprocess.run(
            [
                openface_installed+"FeatureExtraction",
                "-3Dfp",
                "-pose",
                # "-tracked",
                "-f",
                video_dir + speaker + "/" + video,
                "-out_dir",
                openface_dir + speaker
            ]
        )

    if __name__ == "__main__":
        # write face id
        pool = multiprocessing.Pool()
        pool.starmap(
            write_face_id,
            zip([speaker]*len(video_name_list),
                [f.split(".")[0] + ".csv" for f in video_name_list]))
        pool.close()
        pool.join()

print("finished in", time.time()-start_time, "seconds")
exit()
