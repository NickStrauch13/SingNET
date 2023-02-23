import os
import matplotlib.pyplot as plt
#for loading and visualizing audio files
import librosa
import librosa.display
from pydub import AudioSegment
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from PreprocessingFunctions import convert_mp3s_to_wav, get_spectro_from_wav


#audio_clips = os.listdir(audio_fpath)
audio_clips_example = ["Broad-billed_Motmot_105.mp3"]
convert_mp3s_to_wav(audio_clips_example, mp3_path="data/costa_rica/mp3/", export_path="")

get_spectro_from_wav(audio_clips=["Broad-billed_Motmot_105_1.wav"], wav_path="data/costa_rica/wav/", export_path="data/costa_rica/")




