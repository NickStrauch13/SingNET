import numpy as np
import pydub
import requests
import os
import json
import librosa
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt

def mp3_to_np(mp3_file, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(mp3_file)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

    
def np_to_mp3(dest_file, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(dest_file, format="mp3", bitrate="320k")
    

# Finds the max-sum window (sub array) in a given np array.
#
# array = array or list to parse
# sr = sample rate
# samples_per_window = length of desired window 
# stride_coeff = controls stride of sliding window. stride = stride_coeff * sr
#              (In other words, a stride_coeff = .5 means a stride of half a second)
#
# Returns the start (i) and end (j) indicies of the window in the given array\
#
# *** Perhaps in the future we should punish window values near the start and end. This would help center the bird call ***    
def find_max_window(array, sr, samples_per_window, stride_coeff=.5):
    if samples_per_window > len(array):
        return (-1, -1)
    stride = int(stride_coeff * sr)
    if stride > len(array):
        return (-1, 1)
    if isinstance(array, list):
        array = np.array(array)
    array = abs(array)
    max_sum = 0
    best_i, best_j = -1, -1
    i, j = 0, samples_per_window
    while j <= len(array):
        curr_sum = sum(array[i:j])
        if curr_sum > max_sum:
            max_sum = curr_sum
            best_i, best_j = i, j
        i += stride
        j += stride
    return (best_i, best_j)


# Extracts max total magnitude windows "window_size" seconds long.
#
# audio_array = 1D numpy array representing audio file
# sr = sample rate
# window_size = desired length of windows in seconds
#
# Returns a list of numpy arrays (windows)
def extract_best_windows(audio_array, sr, window_size=5):
    samples_per_window = sr * window_size
    if samples_per_window > len(audio_array):
        return []
    ret = []
    start, end = find_max_window(audio_array, sr, samples_per_window)
    max_window = audio_array[start:end]
    ret.append(max_window)
    ret.extend(extract_best_windows(audio_array[0:start], sr, window_size=5))
    ret.extend(extract_best_windows(audio_array[end:], sr, window_size=5))
    return ret


# audio_clips: list or array of desired wav file names  (typically created using os.listdir(directory_path))
# mp3_path: path to diectory containing all desired mp3's to be converted. Should end with /
# export_path: path to directory to which all wav files will be exported to. Should end with /
def convert_mp3s_to_wav(audio_clips, mp3_path, export_path):
    for i in range(len(audio_clips)):
        if audio_clips[i].endswith(".mp3"):
            tempPath = mp3_path + audio_clips[i]
            song = AudioSegment.from_mp3(tempPath)
            song.export(export_path+audio_clips[i].replace("mp3","wav"))
       
    
# audio_clips: list or array of desired wav filepaths  (typically created using os.listdir(directory_path))
# wav_path: path to diectory containing all desired wav's to be used. Should end with /
# export_path: path to directory to which all spectrograms will be exported to. Should end with /
# make_subdirs: Very specific usecase. Will make subdirs useful for training splits if following our naming convention.
def get_spectro_from_wav(audio_clips, wav_path, export_path, make_subdirs=False):
    for i in range(len(audio_clips)):
        if audio_clips[i].endswith(".wav"):
            tempPath = wav_path + audio_clips[i]
            x, sr = librosa.load(tempPath, sr=None)
            # X = librosa.stft(x, n_fft=2048)
            # Xdb = librosa.amplitude_to_db(abs(X))  ## Use These three lines for spectrogram
            # librosa.util.normalize(Xdb)
            Z = librosa.feature.melspectrogram(x,sr)
            Zdb = librosa.amplitude_to_db(abs(Z))  ## Use these three lines for MEL spectrogram
            librosa.util.normalize(Zdb)## I prefer MEL it is significantly faster and seems to give more interesting results
            plt.figure(figsize=(14, 5))
            librosa.display.specshow(Zdb, sr=sr, x_axis='time', y_axis='hz',cmap='viridis')
            plt.title("")
            plt.xlabel("")
            plt.ylabel("")
            plt.axis("off")
            plt.margins(x=0)
            dir_path = export_path
            if make_subdirs:
                dir_name = "_".join(audio_clips[i].split("_")[:-2])  #This line removes the numbers from the end of the filenames
                dir_path = export_path+dir_name
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                dir_path = dir_path + "/"
            plt.savefig(dir_path+audio_clips[i].replace("wav", "jpg"), bbox_inches="tight", pad_inches=0)
            plt.close()
            print(i*1.0/len(audio_clips))