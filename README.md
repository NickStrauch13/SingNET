# Singapore Bird Species Classification using Audio Recordings
## By Nick Strauch and Nick Conterno
This repository contains code for a machine learning model that can identify Singapore bird species based on their calls using audio recordings. The model uses a modified MobileNet architecture to analyze spectrograms generated from input audio files, with spectral gating applied to denoise the files. We also trained a small model on 5 common Costa Rica bird species for inital testing. 

### Dataset
The dataset used for training the model comes from Xeno-Canto, an online database of bird sounds from around the world. The dataset consists of audio recordings in MP3 format, with long recordings divided into a subset of their best windows. The dataset contains recordings of various Singapore bird species, with each recording labeled with the corresponding species name.
