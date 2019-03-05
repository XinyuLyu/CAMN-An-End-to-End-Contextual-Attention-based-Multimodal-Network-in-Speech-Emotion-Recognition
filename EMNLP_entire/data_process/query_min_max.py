from pydub import AudioSegment
import os, re
import numpy as np
import scipy.io
from file_helper import get_file_index
from audio_helper import *

audio_path = "wav/"

min = 0
max = 0
k = 0


for i in range(10038):
    fs, wav_file = wavfile.read(audio_path + str(i) + '.wav')
    # if min > float(wav_file.min()):
    #     min = float(wav_file.min())
    # if max < float(wav_file.max()):
    #     max = float(wav_file.max())
    if wav_file.min() == -32768:
        k += 1
        print(str(i) + '.wav detected')

print(str(k))