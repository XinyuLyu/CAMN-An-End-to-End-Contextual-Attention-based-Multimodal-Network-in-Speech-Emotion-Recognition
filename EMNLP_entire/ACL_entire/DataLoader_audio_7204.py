from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from word2vec import embed_onehot, get_dictionary, initial_embed
import numpy as np
import random
import string
import scipy.io as scio
from sklearn.utils import shuffle
from data import load_data_audio_attention, save_data_audio_attention
from augmentation import shuffle_text, get_dictionary, get_keys, get_text, get_index, drop, replace_synonym
import copy

label_category = ['ang', 'exc', 'sad', 'fru', 'hap', 'neu']
dic_path = r'E:/Yue/Entire Data/ACL_2018_entire/dictionary_new.txt'
label_path = r'E:/Yue/Entire Data/ACL_2018_entire/label_output_new.txt'
audio_path = r'E:/Yue/Entire Data/ACL_2018_entire/Word_Mat_New_1/'
text_path = r'E:/Yue/Entire Data/ACL_2018_entire/text_output_new.txt'
embed_path = r'E:/Yue/Entire Data/ACL_2018_entire/'
maxlen = 50
numclass = 4
num = 7204
batch_size = 4


def get_label(path):
    f = open(path, 'r')
    #             0         1         2         3         1         3
    statistic = {'ang': 0, 'exc': 0, 'sad': 0, 'fru': 0, 'hap': 0, 'neu': 0}
    res = []
    for line in f:
        if line.split()[0] == label_category[0]:
            statistic[label_category[0]] += 1
            res.append(0)
        elif line.split()[0] == label_category[1]:
            statistic[label_category[1]] += 1
            res.append(1)
        elif line.split()[0] == label_category[2]:
            statistic[label_category[2]] += 1
            res.append(2)
        elif line.split()[0] == label_category[3]:
            statistic[label_category[3]] += 1
            res.append(3)
        elif line.split()[0] == label_category[4]:
            statistic[label_category[4]] += 1
            res.append(1)
        elif line.split()[0] == label_category[5]:
            statistic[label_category[5]] += 1
            res.append(3)
    print(statistic)
    return res


def get_mat_data(path):
    res = []
    i = 0
    while i < 7204:
        tmp = scio.loadmat(path + str(i) + ".mat")
        tmp = tmp['z1']
        tmp = sequence.pad_sequences(tmp, padding='post', truncating='post', dtype='float32', maxlen=maxlen)
        tmp = tmp.transpose()
        res.append(tmp)
        i += 1
    return res


def get_text_data(path, dic):
    f = open(path, 'r')
    res = []
    i = 0
    for line in f:
        text = embed_onehot(dic, line.translate(str.maketrans('', '', string.punctuation)))
        res.append(text)
        i += 1
    f.close()
    return res


def seprate_by_emotion(path, data):
    f = open(path, 'r')
    ang = []
    hap_exc = []
    sad = []
    fru_neu = []
    # neu = []
    i = 0
    for line in f:
        if line.split()[0] == label_category[0]:
            ang.append(data[i])
        elif line.split()[0] == label_category[1]:
            hap_exc.append(data[i])
        elif line.split()[0] == label_category[2]:
            sad.append(data[i])
        elif line.split()[0] == label_category[3]:
            fru_neu.append(data[i])
        elif line.split()[0] == label_category[4]:
            hap_exc.append(data[i])
        elif line.split()[0] == label_category[5]:
            fru_neu.append(data[i])
        i += 1
    return ang, hap_exc, sad, fru_neu


def seperate_dataset(audio_data, text_data, label):
    train_text_data, train_audio_data, test_text_data, test_audio_data = [], [], [], []
    train_label, test_label = [], []
    ang_text, hap_exc_text, sad_text, fru_neu_text = seprate_by_emotion(label_path, text_data)
    #ang_text_shuffle = copy.deepcopy(ang_text)
    #shuffle_text(ang_text_shuffle)
    #hap_exc_text_shuffle = copy.deepcopy(hap_exc_text)
    #shuffle_text(hap_exc_text_shuffle)
    #sad_text_shuffle = copy.deepcopy(sad_text)
    #shuffle_text(sad_text_shuffle)
    # dict = get_dictionary(r'E:/Yue/Entire Data/ACL_2018_entire/dictionary_new.txt')
    #ang_text_drop=drop(ang_text,dict,0.4)
    #hap_exc_text_drop=drop(hap_exc_text,dict,0.4)
    #sad_text_drop=drop(sad_text,dict,0.4)
    # ang_text_replace = replace_synonym(ang_text,dict)
    # hap_exc_text_replace = replace_synonym(hap_exc_text,dict)
    # sad_text_replace = replace_synonym(sad_text,dict)
    ang_audio, hap_exc_audio, sad_audio, fru_neu_audio = seprate_by_emotion(label_path, audio_data)
    ang_label, hap_exc_label, sad_label, fru_neu_label = seprate_by_emotion(label_path, label)
    ang_i = 0
    while ang_i < len(ang_audio):
        if random.randint(0, 100) < 80:
            train_text_data.append(ang_text[ang_i])
            #train_text_data.append(ang_text_shuffle[ang_i])
            #train_text_data.append(ang_text_drop[ang_i])
            # train_text_data.append(ang_text_replace[ang_i])
            train_audio_data.append(ang_audio[ang_i])
            train_label.append(ang_label[ang_i])
            # train_label.append(ang_label[ang_i])
        else:
            test_text_data.append(ang_text[ang_i])
            test_audio_data.append(ang_audio[ang_i])
            test_label.append(ang_label[ang_i])
        ang_i += 1

    hap_exc_i = 0
    while hap_exc_i < len(hap_exc_audio):
        if random.randint(0, 100) < 80:
            train_text_data.append(hap_exc_text[hap_exc_i])
            #rain_text_data.append(hap_exc_text_shuffle[hap_exc_i])
            #train_text_data.append(hap_exc_text_drop[hap_exc_i])
            # train_text_data.append(hap_exc_text_replace[hap_exc_i])
            train_audio_data.append(hap_exc_audio[hap_exc_i])
            train_label.append(hap_exc_label[hap_exc_i])
            # train_label.append(hap_exc_label[hap_exc_i])
        else:
            test_text_data.append(hap_exc_text[hap_exc_i])
            test_audio_data.append(hap_exc_audio[hap_exc_i])
            test_label.append(hap_exc_label[hap_exc_i])
        hap_exc_i += 1

    sad_i = 0
    while sad_i < len(sad_audio):
        if random.randint(0, 100) < 80:
            train_text_data.append(sad_text[sad_i])
            #train_text_data.append(sad_text_shuffle[sad_i])
            #train_text_data.append(sad_text_drop[sad_i])
            # train_text_data.append(sad_text_replace[sad_i])
            train_audio_data.append(sad_audio[sad_i])
            train_label.append(sad_label[sad_i])
            # train_label.append(sad_label[sad_i])
        else:
            test_text_data.append(sad_text[sad_i])
            test_audio_data.append(sad_audio[sad_i])
            test_label.append(sad_label[sad_i])
        sad_i += 1

    fru_neu_i = 0
    while fru_neu_i < len(fru_neu_audio):
        # ang data
        if random.randint(0, 100) < 80:
            train_text_data.append(fru_neu_text[fru_neu_i])
            train_audio_data.append(fru_neu_audio[fru_neu_i])
            train_label.append(fru_neu_label[fru_neu_i])

        else:
            test_text_data.append(fru_neu_text[fru_neu_i])
            test_audio_data.append(fru_neu_audio[fru_neu_i])
            test_label.append(fru_neu_label[fru_neu_i])
        fru_neu_i += 1

    train_text_data, train_label = shuffle(train_text_data, train_label)
    test_text_data, test_label = shuffle(test_text_data, test_label)
    return train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label


def analyze_data(test_label, result):
    r_0 = {'0': 0, '1': 0, '2': 0, '3': 0}
    r_1 = {'0': 0, '1': 0, '2': 0, '3': 0}
    r_2 = {'0': 0, '1': 0, '2': 0, '3': 0}
    r_3 = {'0': 0, '1': 0, '2': 0, '3': 0}

    i = 0
    while i < len(test_label):  # 4
        if test_label[i] == 0:
            r_0[str(result[i])] += 1
        elif test_label[i] == 1:
            r_1[str(result[i])] += 1
        elif test_label[i] == 2:
            r_2[str(result[i])] += 1
        elif test_label[i] == 3:
            r_3[str(result[i])] += 1

        i += 1
    return r_0, r_1, r_2, r_3


def data_generator(path, audio_data, audio_label, num):
    i = 0
    while 1:
        res, res_label = [], []
        j = 0
        while j < batch_size:
            if i == num:
                i = 0
            tmp = scio.loadmat(path + str(audio_data[i]) + ".mat")
            tmp = tmp['z1']
            res.append(tmp)
            res_label.append(audio_label[i])
            j += 1
            i += 1
        res = sequence.pad_sequences(res, padding='post', truncating='post', dtype='float32', maxlen=maxlen)
        yield ({'input_2': np.array(res)}, {'dense_2': np.array(res_label)})
        # ({'input_2': np.array(res_a), 'input_3': np.array(res_t)}, {'dense_1': np.array(res_label)})


def data_generator_output(path, audio_data, audio_label, num):
    i = 0
    while 1:
        res, res_label = [], []
        if i == num:
            i = 0
        tmp = scio.loadmat(path + str(audio_data[i]) + ".mat")
        tmp = tmp['z1']
        res.append(tmp)
        res_label.append(audio_label[i])
        i += 1
        res = sequence.pad_sequences(res, padding='post', truncating='post', dtype='float32', maxlen=maxlen)
        yield (np.array(res), np.array(res_label))


def get_hier_mat_data():
    res = []
    i = 0
    while i < num:
        res.append(i)
        i += 1
    return res


def get_data():
    dic = get_dictionary(dic_path)
    embed_matrix = initial_embed(dic, embed_path)
    label = get_label(label_path)
    audio_data = get_hier_mat_data()
    text_data = get_text_data(text_path, dic)
    train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label_o = seperate_dataset(
        audio_data, text_data, label)
    train_label = to_categorical(train_label, num_classes=numclass)
    train_text_data = sequence.pad_sequences(train_text_data, padding='post', truncating='post', maxlen=maxlen)
    test_label = to_categorical(test_label_o, num_classes=numclass)
    test_text_data = sequence.pad_sequences(test_text_data, padding='post', truncating='post', maxlen=maxlen)
    #save_data_audio_attention(train_audio_data, train_text_data, train_label, test_audio_data, test_text_data,
    #                          test_label, test_label_o)
    return train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic


def get_data1():
    dic = get_dictionary(dic_path)
    embed_matrix = initial_embed(dic, embed_path)
    train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o = load_data_audio_attention()
    return train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic