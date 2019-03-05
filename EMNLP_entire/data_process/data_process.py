import os
import numpy as np
import scipy.io
from scipy.io import wavfile
from file_helper import get_file_index
from audio_helper import *

word_path = "word-alignment"
audio_path = "wav/"
word_files = os.listdir(word_path)

max = 0

# 遍历word_alignment文件夹
for file in word_files:
    if not os.path.isdir(file):
        # 建立np数组
        array = []

        # 读取word_alignment文件
        f = open(word_path + "/" + file);
        line = f.readline()

        # 得到文件名(数字)
        file_index = get_file_index(f.name)

        # 读取对应的.wav文件
        fs, audio_file = wavfile.read(audio_path + file_index +'.wav')

        # normalize
        amin = audio_file.min()
        amax = audio_file.max()

        audio_file = [(float(i) + 32768) / 65535 for i in audio_file]

        # audio_file = normalize(audio_file)

        # 每一行读取，找到duration
        while line != "":
            s = line.split(" ")
            # 得到每一行的duration
            end_time = float(s[1])
            start_time = float(s[0])
            # 得到每一个词的数据
            array_append = get_segment(start_time, end_time, audio_file)
            array.append(array_append)
            # np.concatenate(array, array_append)

            line = f.readline()
        f.close()
        print(f.name + ' finished')
        # 归一化
        # array = normalize(array)
        scipy.io.savemat('result/' + file_index+'.mat', {'data': array})

