import os
import subprocess
import pandas as pd
import time


openface_installed = "/home/alkhemi/OpenFace/build/bin/"
video_dir = "../video_smallset_webm/"
openface_dir = "../test/"

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
    df.to_csv(openface_dir+speaker+"/" + openface_filename, index=False)


start_time = time.time()

print("indexing data")
video_ext = ["mp4","wmv","mkv","webm","mpg","avi"]
file_index = []
for root, _, files in os.walk(video_dir):
    files.sort()
    for file in files:
        for ext in video_ext:
            if file.endswith(ext):
                file_index.append([root.split(video_dir)[-1], file])

file_index.sort()
# print("getting openface data")
for file in file_index[0:1]:
    source_path = os.path.join(video_dir,file[0],file[1]) 
    target_path = os.path.join(openface_dir,file[0]) 
    if not os.path.exists(target_path):
        os.makedirs(target_path)    
    subprocess.run(
                [
                    openface_installed+"FeatureExtraction",
                    "-3Dfp",
                    "-pose",
                    # "-tracked",
                    "-f",
                    source_path,
                    "-out_dir",
                    target_path
                ]
            )
    df = pd.read_csv(os.path.join(target_path,file[1]), skipinitialspace=True)
    df.loc[:, "face_id"] = file[0].spli('/')[-1]
    df.loc[:, "pose_Rx"] = df.loc[:, "pose_Rx"].round(3)
    df.loc[:, "pose_Ry"] = df.loc[:, "pose_Ry"].round(3)
    df.loc[:, "pose_Rz"] = df.loc[:, "pose_Rz"].round(3)
    df.to_csv(os.path.join(target_path,file[1]), index=False)


print("finished in", time.time()-start_time, "seconds")
exit()
