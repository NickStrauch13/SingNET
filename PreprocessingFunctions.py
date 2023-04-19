import numpy as np
import pydub
import requests
import os
import json
import librosa
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt
from PIL import Image

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
    
    
def convert_wav_to_mp3(wav_file_path, export_path):
    song = AudioSegment.from_file(wav_file_path)
    song.export(export_path, format="mp3")


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
        try:
            curr_sum = sum(array[i:j])
        except:
            curr_sum = np.inf
        if curr_sum > max_sum:
            max_sum = curr_sum
            best_i, best_j = i, j
        i += stride
        j += stride
    return (best_i, best_j), max_sum


# Extracts max total magnitude windows "window_size" seconds long.
#
# audio_array = 1D numpy array representing audio file
# sr = sample rate
# window_size = desired length of windows in seconds
#
# Returns a list of numpy arrays (windows)
def extract_best_windows(audio_array, sr, max_power, window_size=5, depth=0):
    samples_per_window = sr * window_size
    if samples_per_window > len(audio_array):
        return []
    ret = []
    (start, end), power = find_max_window(audio_array, sr, samples_per_window)
    if power > max_power:
        max_power = power
    max_window = audio_array[start:end]
    if power > .4*max_power:
        ret.append(max_window)
    if depth > 200:
        return ret
    ret.extend(extract_best_windows(audio_array[0:start], sr, max_power=max_power, window_size=5, depth=depth+1))
    if depth > 200:
        return ret
    ret.extend(extract_best_windows(audio_array[end:], sr, max_power=max_power, window_size=5, depth=depth+1))
    return ret


# audio_clips: list or array of desired mp3 file names  (typically created using os.listdir(directory_path))
# mp3_path: path to diectory containing all desired mp3's to be converted. Should end with /
# export_path: path to directory to which all wav files will be exported to. Should end with /
def convert_mp3s_to_wav(audio_clips, mp3_path, export_path):
    for i in range(len(audio_clips)):
        if audio_clips[i].endswith(".mp3"):
            try:
                tempPath = mp3_path + audio_clips[i]
                song = AudioSegment.from_mp3(tempPath)
                song.export(export_path+audio_clips[i].replace("mp3","wav"))
            except:
                print(f"{audio_clips[i]} failed to convert")
       
    
# audio_clips: list or array of desired wav filepaths  (typically created using os.listdir(directory_path))
# wav_path: path to diectory containing all desired wav's to be used. Should end with /
# export_path: path to directory to which all spectrograms will be exported to. Should end with /
# make_subdirs: Very specific usecase. Will make subdirs useful for training splits if following our naming convention.
def get_spectro_from_wav(audio_clips, wav_path, export_path, make_subdirs=False):
    num_files = len(audio_clips)
    c=0
    for i in range(num_files):
        if audio_clips[i].endswith(".wav"):
            try:
                tempPath = wav_path + audio_clips[i]
                x, sr = librosa.load(tempPath, sr=None)
                # X = librosa.stft(x, n_fft=2048)
                # Xdb = librosa.amplitude_to_db(abs(X))  ## Use These three lines for spectrogram
                # librosa.util.normalize(Xdb)
                Z = librosa.feature.melspectrogram(y=x,sr=sr)
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
                c+=1
                if c % 250 == 0:
                    print(round(c/num_files, 5))
            except:
                print(f"Failed to make spectrogram for {audio_clips[i]}")

            
def wav_to_np(wav_file, normalized=False):  # Sus af
    """WAV to numpy array"""
    a = pydub.AudioSegment.from_wav(wav_file)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y
    
def np_to_wav(dest_file, sr, x, normalized=False):
    """numpy array to WAV"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(dest_file, format="wav", bitrate="320k")
    

# wav_file_path: path to wav file
# export_path: path to desired export directory. For both the wav windows and the spectrograms. Should end with "/"
# file_name_root: the name of your desired saved files. They will be saved like "file_name_root_27.wav" or "file_name_root_33.jpg"
# window_size: length of extracted windows in seconds
def convert_long_wav_to_spectro_windows(wav_file_path, export_path, file_name_root, window_size=5):
    convert_wav_to_mp3(wav_file_path, export_path+"mp3version.mp3")
    sr, audio_array = mp3_to_np(export_path+"mp3version.mp3")
    if len(audio_array.shape) == 2:
        audio_array = np.mean(audio_array, axis=1)
    windows = extract_best_windows(audio_array, sr, window_size)
    if not os.path.exists(export_path+"wav"):
        os.mkdir(export_path+"wav")
    if not os.path.exists(export_path+"spectrogram"):
        os.mkdir(export_path+"spectrogram")
        
    for i in range(0, len(windows)):
        full_export_path = export_path + "wav/" + file_name_root + f"_{i}.wav"
        np_to_wav(full_export_path, sr, windows[i])
    
    audio_clips = os.listdir(export_path + "wav")
    get_spectro_from_wav(audio_clips, export_path+"wav/", export_path+"spectrogram/")
    

# Culls a dataset 
#
# original_data_path: path to original dataset directory
# new_data_path: path to new dataset directory
# max_per_class: maximum number of samples per class after culling
def reduce_dataset(original_data_path, new_data_path, max_per_class=30):
    # Create the new directory if it doesn't exist
    if not os.path.exists(new_data_path):
        os.mkdir(new_data_path)

    # Loop through each subdirectory in the original directory
    for subdir in os.listdir(original_data_path):
        subdir_path = os.path.join(original_data_path, subdir)
        # Create the corresponding subdirectory in the new directory
        new_subdir_path = os.path.join(new_data_path, subdir)
        if not os.path.exists(new_subdir_path):
            os.mkdir(new_subdir_path)
        # Get a list of all the jpg files in the subdirectory
        jpg_files = [f for f in os.listdir(subdir_path) if f.endswith('.jpg')]
        # Choose a random subset of the jpg files
        num_files_to_copy = min(len(jpg_files), max_per_class) 
        files_to_copy = random.sample(jpg_files, num_files_to_copy)
        # Copy the selected files to the corresponding subdirectory in the new directory
        for file in files_to_copy:
            src_path = os.path.join(subdir_path, file)
            dst_path = os.path.join(new_subdir_path, file)
            shutil.copy(src_path, dst_path)
            

# compute some statistics about a dataset
#
#dataloader: dataset
def compute_class_stats(dataloader):
    class_samples = {}
    total_samples = 0
    for _, labels in dataloader:
        total_samples += len(labels)
        for label in labels:
            if label.item() not in class_samples:
                class_samples[label.item()] = 1
            else:
                class_samples[label.item()] += 1

    num_classes = len(class_samples)
    class_samples = [class_samples[i] for i in range(num_classes)]

    return num_classes, total_samples, class_samples


# Reduce the resolution of an image down to 244x244
#
# input_dir: direcotry with subdirs of jpg's
# output_dir: output directory for 244x244 images
# min_num_files: minimum number of files in a sub-directory for the downsampling to be applied
def downsample_images(input_dir, output_dir, min_num_files=-1):
    # Recursively iterate over all subdirs in the input directory
    for root, dirs, files in os.walk(input_dir):
        if len(files) >= min_num_files:
            for file in files:
                # Check if the file is a JPEG image
                if file.endswith('.jpg'):
                    # Construct the input and output file paths
                    input_path = os.path.join(root, file)
                    output_path = input_path.replace(input_dir, output_dir)

                    # Create the output directory if it doesn't exist
                    output_subdir = os.path.dirname(output_path)
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    # Load the image and downsample it
                    img = Image.open(input_path)
                    img_array = np.array(img)
                    block_size = (img_array.shape[0] // 244, img_array.shape[1] // 244)
                    img_array = img_array[:244 * block_size[0], :244 * block_size[1]]
                    img_array = img_array.reshape((244, block_size[0], 244, block_size[1], 3))
                    img_array = np.mean(img_array, axis=(1, 3)).astype(np.uint8)
                    img = Image.fromarray(img_array)

                    # Save the downsampled image
                    img.save(output_path)