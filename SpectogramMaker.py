import os
import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display

#to play audio
import IPython.display as ipd

audio_fpath = "data/mp3/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
## ya this needs to be accurate

for i in range(len(audio_clips)):
    # print(type(x), type(sr))
    # print(x.shape, sr)
    print(audio_clips[i])
    x, sr = librosa.load(audio_fpath + audio_clips[i], sr=None)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.savefig("data/spectrogramImages/"+audio_clips[i].replace("mp3", "jpg"))