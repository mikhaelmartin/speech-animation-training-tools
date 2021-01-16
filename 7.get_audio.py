# temp edited

import os
import multiprocessing
import moviepy.editor
from scipy.signal import butter, sosfilt
from scipy.io import wavfile
import time
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import sosfreqz

# video_dir = "../video/"
video_dir = "../video/"
audio_dir = "../audio/"
audio_filtered_dir = "../audio_filtered/"

if not os.path.exists(audio_dir):
    os.mkdir(audio_dir)

if not os.path.exists(audio_filtered_dir):
    os.mkdir(audio_filtered_dir)

# bikin filter
sos = butter(13, [60, 8000], output='sos', btype='bandpass', fs=44100)

# # plot frequency respones
# w, h = sosfreqz(sos, worN=512, fs=44100)
# h_db = 20*np.log10(abs(h))
# plt.plot(w, h_db)
# plt.title('Frequency Response')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Gain (dB)')
# plt.ylim(-200, 5)
# # plt.yscale('log')
# plt.grid('on')
# plt.show()


def wav_extract(speaker, video):
    try:
        audioclip = moviepy.editor.AudioFileClip(
            video_dir + speaker + "/" + video, fps=44100)
        audio_array = audioclip.to_soundarray(fps=44100)[:, 0]
        fs = audioclip.fps
        audioclip.close

        # # save raw audio wav
        # wavfile.write(
        #     audio_dir + speaker + "/" + video.split(".")[-2] +
        #     ".wav", rate=fs, data=audio_array)

        # filter audio
        audio_array = sosfilt(sos, audio_array)

        # save filtered audio wav
        wavfile.write(
            audio_filtered_dir + speaker + "/" + video.split(".")[-2] +
            ".wav", rate=fs, data=audio_array)
    except Exception as e:
        raise Exception(e.message)


# video is put inside folder for each speaker
speaker_list = [
    file for file in os.listdir(video_dir) if os.path.isdir(video_dir + file)
]
speaker_list.sort()


print("getting audio")
if __name__ == "__main__":
    start_time = time.time()
    for speaker in speaker_list:

        print("processing " + speaker)
        # if not os.path.exists(audio_dir + speaker):
        #     os.mkdir(audio_dir + speaker)
        if not os.path.exists(audio_filtered_dir + speaker):
            os.mkdir(audio_filtered_dir + speaker)

        video_name_list = [
            item for item in os.listdir(video_dir + speaker)
            if (item.split(".")[-1] == "mp4")
            or (item.split(".")[-1] == "wmv")
            or (item.split(".")[-1] == "avi")
            or (item.split(".")[-1] == "webm")
            or (item.split(".")[-1] == "mpg")]

        pool = multiprocessing.Pool(10)

        # dari video ambil audio taro ke folder audio
        try:
            pool.starmap(
                wav_extract,
                zip([speaker]*len(video_name_list), video_name_list)

            )
        except Exception as e:
            print(e.message)
            pool.close()
            pool.join()
        pool.close()
        pool.join()
    print("finished in", time.time()-start_time, "seconds")
    exit()
