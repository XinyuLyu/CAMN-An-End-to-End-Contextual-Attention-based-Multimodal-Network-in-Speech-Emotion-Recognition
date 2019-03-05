import librosa
import numpy as np
import os
import scipy.io
from file_helper import get_file_index_stft

rootdir = 'test/'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        relative_path = os.path.join(subdir, file)
        wav_file, sr = librosa.load(relative_path)

        vec = np.abs(librosa.stft(wav_file, n_fft=1024))

        file_index = file[0: len(file) - 4]

        scipy.io.savemat(subdir + '/' + file_index + '.mat', {'data': vec})
