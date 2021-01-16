import os
import multiprocessing
import pandas as pd
import time

minimum_frame_length = 20
minimum_confidence = 0.85

openface_dir = "../openface/"
openface_invalid_dir = "../openface_invalid/"
video_dir = "../video/"
video_invalid_dir = "../video_invalid/"
alignment_dir = "../alignment/"
alignment_invalid_dir = "../alignment_invalid/"

for dir in [openface_dir, video_dir, alignment_dir]:
    if not os.path.exists(dir):
        print(dir, "not found. exiting..")
        exit()
for dir in [openface_invalid_dir, video_invalid_dir, alignment_invalid_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)

speaker_list = [item for item in os.listdir(
    openface_dir) if os.path.isdir(openface_dir + item)]
speaker_list.sort()


def validate(speaker, filename):
    df = pd.read_csv(openface_dir+speaker+"/"+filename, skipinitialspace=True)

    # validate length
    if any(v < minimum_confidence for v in df["confidence"].values) or \
            (0 in df["success"].values) or len(df) < minimum_frame_length:

        # move openface file
        os.rename(openface_dir+speaker+"/"+filename,
                  openface_invalid_dir+speaker+"/"+filename)

        # move openface details file
        of_detail_filename = filename[:-4]+"_of_details.txt"
        if os.path.exists(openface_dir+speaker+"/"+of_detail_filename):
            os.rename(openface_dir+speaker+"/"+of_detail_filename,
                      openface_invalid_dir+speaker + "/"+of_detail_filename)

        # move video file
        video_filename = filename[:-4]+".mpg"
        if os.path.exists(video_dir+speaker+"/"+video_filename):
            os.rename(video_dir+speaker+"/"+video_filename,
                      video_invalid_dir+speaker+"/"+video_filename)
        else:
            print("warning: "+video_dir+speaker +
                  "/"+video_filename+" not exist")

        # move alignment file
        alignment_filename = filename[:-4]+".align"
        if os.path.exists(alignment_dir+speaker+"/"+alignment_filename):
            os.rename(alignment_dir+speaker+"/"+alignment_filename,
                      alignment_invalid_dir+speaker+"/"+alignment_filename)
        else:
            print("warning: "+alignment_dir+speaker +
                  "/"+alignment_filename+" not exist")

    return None


if __name__ == "__main__":
    start_time = time.time()
    print("validate video using openface files")
    for speaker in speaker_list:
        print("Processing " + speaker)

        for dir in [openface_invalid_dir, video_invalid_dir,
                    alignment_invalid_dir]:
            if not os.path.exists(dir + speaker):
                os.mkdir(dir+speaker)

        openface_list = [item for item in os.listdir(
            openface_dir+speaker) if item[-4:] == ".csv"]

        pool = multiprocessing.Pool()

        # dari openface ambil landmark taro di folder landmark
        pool.starmap(
            validate,
            zip([speaker]*len(openface_list), openface_list)
            # zip([speaker]*1, ["prwp7p.csv"])
        )
        pool.close()
        pool.join()
    print("finished in", time.time()-start_time, "seconds")
    exit()
