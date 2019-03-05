from pydub import AudioSegment
import os, re
import numpy as np
import scipy.io
from file_helper import get_file_index
from audio_helper import *

word_path = "word-alignment"
audio_path = "wav/"
word_files = os.listdir(word_path)

max = 0

# 遍历word_alignment文件夹
for file in word_files:
    if not os.path.isdir(file):
        # 读取word_alignment文件
        f = open(word_path + "/" + file);
        line = f.readline()

        # 得到文件名(数字)
        file_index = get_file_index(f.name)

        # 建立子文件夹
        current_path = 'result/' + file_index + '/'

        folder = os.path.exists(current_path)

        if not folder:
            os.makedirs(current_path)

        # 读取对应的.wav文件
        wav_file = AudioSegment.from_file(audio_path + file_index + '.wav', format="wav")
        print(wav_file.duration_seconds)
        print(len(wav_file))

        # audio_file = normalize(audio_file)

        # 每一行读取，找到duration
        i = 0
        while line != "":
            s = line.split(" ")
            # 得到每一行的duration
            end_time = float(s[1])
            start_time = float(s[0])
            duration = end_time - start_time
            if duration < 0.04:
                wav_file[start_time * 1000:(start_time + 0.04) * 1000]. \
                    export(current_path + str(i) + '.wav', format='wav')
            else:
                wav_file[start_time * 1000:end_time * 1000]. \
                    export(current_path + str(i) + '.wav', format='wav')
            i += 1
            line = f.readline()

        f.close()
        print(f.name + ' finished')
        # 归一化
