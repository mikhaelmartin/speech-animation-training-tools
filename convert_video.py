import subprocess
import os


source_dir = "/home/alkhemi/Documents/thesis/video_smallset/"
result_dir = "/home/alkhemi/Documents/thesis/video_smallset_webm/"
source_dir = "/home/alkhemi/Documents/source/"
result_dir = "/home/alkhemi/Documents/target/"


for root, dirs, files in os.walk(source_dir):
    files.sort()
    for file in files:
        target_dir = result_dir
        for dir in root.split(source_dir)[-1].split("/"):
            target_dir = os.path.join(target_dir, dir)
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)
        if file.endswith("mpg") or file.endswith("mp4") or file.endswith("mkv"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    os.path.join(root, file),
                    "-c:v",
                    "vp8",
                    "-c:a",
                    "libvorbis",
                    os.path.join(target_dir, file.split(".")[0] + ".webm"),
                ]
            )
