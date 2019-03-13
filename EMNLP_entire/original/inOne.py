import scipy.io as sio
import numpy as np
import os
import math
from keras.preprocessing import sequence

path_to_mat = 'E:\\Yue\\Entire Data\\iemocap_ACMMM_2018\\IEMOCAP_Mat_64'
path_to_rule = 'E:\\Yue\\Entire Data\\iemocap_ACMMM_2018\\word-alignment'
output_path = 'E:\\Yue\\Entire Data\\iemocap_ACMMM_2018\\Word_Mat_64'
#audio_path_train = 'E:\\Yue\\Entire Data\\iemocap_ACMMM_2018\\IEMOCAP_align\\Word_Mat_Nor_std_40\\'
#output_path_train = 'E:\\Yue\\Entire Data\\iemocap_ACMMM_2018\\Processed_data\\IEMOCAP_Mat_Nor_Align_std_200_40\\'
audio_path_train = 'E:\\Yue\\Entire Data\\iemocap_ACMMM_2018\\IEMOCAP_align\\result_wav\\'
output_path_train = 'E:\\Yue\\Entire Data\\iemocap_ACMMM_2018\\Processed_data\\IEMOCAP_Mat_Whole_Nor_Align_wav_1\\'


def get_rules(rule):
    # get start & end frame of each word -- tuple
    # save start & end frame of all words as a list
    rules = []
    for line in rule:
        start, end, word = line.split(' ')
        start = float(start) * 100
        end = float(end) * 100
        start = int(math.floor(start))
        end = int(math.ceil(end))
        rules.append((start, end))
    return rules


def get_fragments(array, rules, num, filename):
    sum = 0
    for i, rule in enumerate(rules):
        i = str(i)
        start, end = rule
        new_array = array[:, start:end+1]
        if new_array.shape[0] == 0 or new_array.shape[1] == 0:
            zeroLog = open('zeroLog.txt', 'a')
            zeroLog.write(filename + 'Potential empty mat error\n')
            zeroLog.close()
        else:
            print('Shape after converting:', new_array.shape)
        sum += (new_array.shape)[1]
        sio.savemat(output_path+'\\'+str(num)+'\\'+i, mdict={'z1':new_array})
    print(sum)


def get_mat_data(path, num):
    res = []
    i = 0
    while i < num:
        files = os.listdir(path + str(i) + '\\')
        sent = []
        for file in files:
            tmp = sio.loadmat(path + str(i) + '\\' + file)
            tmp = tmp['data']
            tmp = sequence.pad_sequences(tmp, dtype='float32', maxlen=200)
            # print(tmp.shape)
            tmp = tmp.transpose()
            sent.append(tmp)
        print(np.array(sent).shape)
        res.append(sent)
        i += 1
    return res


def output_mat_data(path, res, num):
    i = 0
    while i < num:
        sio.savemat(path+str(i)+".mat", {'z1': res[i]})
        i += 1


# # cut into to words
# for filename in os.listdir(path_to_mat):
#     # load file and gather information
#     num, suffix = filename.split('.')
#     mat_file = path_to_mat + '\\' + filename
#     print('Working on', filename)
#     try:
#         mat = sio.loadmat(mat_file)
#     except TypeError:
#         print('Error load mat file:', filename)
#         logfile = open('errLog.txt', 'a')
#         logfile.write('Error load ' + filename + '\n')
#         logfile.close()
#     array = None
#     for k in mat.keys():
#         if not k.startswith('__'):
#             array = mat.get(k)	 # ndarray (64, n)
#             shape = array.shape
#             print('Shape before converting:', shape)
#     rule = open(path_to_rule+'\\'+num+'.txt', 'r')
#     rules = get_rules(rule)
#
#     # cut sentence level mat file into word level mat file
#     os.mkdir(output_path+'\\'+str(num))
#     get_fragments(array, rules, num, filename)

res1 = get_mat_data(audio_path_train, 10039)
output_mat_data(output_path_train, res1, 10039)
