import os
import matplotlib.pyplot as plt
#for loading and visualizing audio files
import librosa
import librosa.display
from pydub import AudioSegment
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def convertmp3sToWav(audio_clips):
    audio_fpath = "data/costa_rica/mp3_windows/"
    audio_clips = os.listdir(audio_fpath)
    print("No. of .wav files in audio folder = ", len(audio_clips))
    for i in range(len(audio_clips)):
        # print(type(x), type(sr))
        # print(x.shape, sr)
        print(audio_clips[i])
        if not audio_clips[i].endswith(".ipynb_checkpoints"):
            tempPath = audio_fpath + audio_clips[i]
            song = AudioSegment.from_mp3(tempPath)
            song.export("data/costa_rica/wav/"+audio_clips[i].replace("mp3","wav"))
def getSpectroFromWav(audio_clips):
    for i in range(len(audio_clips)):
        # print(type(x), type(sr))
        # print(x.shape, sr)
        print(audio_clips[i])
        if not audio_clips[i].endswith(".ipynb_checkpoints"):
            tempPath = audio_fpath + audio_clips[i]
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
            #plt.colorbar()
            plt.savefig("data/costa_rica/spectrogram/"+audio_clips[i].replace("wav", "jpg"), bbox_inches="tight", pad_inches=0)
            plt.close()
            print(i*1.0/len(audio_clips))
# audio_fpath = "data/costa_rica/wav/" #use this line for convertmp3sToWav
audio_fpath = "data/costa_rica/wav/"  #use this line for getSpectroFromWav

audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
t= time.time()
#convertmp3sToWav(audio_clips)
getSpectroFromWav(audio_clips)

print(time.time()-t)



