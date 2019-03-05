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
res01 = []
res12 = []
res23 = []
res34 = []
res45 = []
res56 = []
res67 = []
res78 = []
res89 = []
res90 = []

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

        # 每一行读取，找到duration
        while line != "":
            s = line.split(" ")
            # 得到每一行的duration
            end_time = float(s[1])
            start_time = float(s[0])
            duration = end_time - start_time

            if 0 < duration < 0.01:
                res01.append(file_index)
            elif 0.01 < duration < 0.02:
                res12.append(file_index)
            elif 0.02 < duration < 0.03:
                res23.append(file_index)
            elif 0.03 < duration < 0.04:
                res34.append(file_index)
            elif 0.04 < duration < 0.05:
                res45.append(file_index)
            elif 0.05 < duration < 0.06:
                res56.append(file_index)
            elif 0.06 < duration < 0.07:
                res67.append(file_index)
            elif 0.07 < duration < 0.08:
                res78.append(file_index)
            elif 0.08 < duration < 0.09:
                res89.append(file_index)
            elif 0.09 < duration < 0.1:
                res90.append(file_index)
            line = f.readline()

        f.close()
        print(f.name + ' finished')

        # 归一化
        # array = normalize(array)
for i in range(9):
    name = 'res' + str(i) + str(i + 1)
    print(str(i) + ' size is ' + str(len(locals()[name])))
print(str(i) + ' size is ' + str(len(res90)))


