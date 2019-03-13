from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from word2vec import embed_onehot, get_dictionary, initial_embed
import numpy as np
import random
import string
import scipy.io as scio
from sklearn.utils import shuffle
from data import load_data_audio, save_data_audio, save_data_audio_wav, load_data_audio_wav, \
    save_data_audio_mfsc_sentence, load_data_audio_mfsc_sentence, load_data_audio_opensmile_sentence, \
    save_data_audio_opensimle_sentence

label_category = ['ang', 'exc', 'sad', 'fru', 'hap', 'neu', 'oth', 'sur', 'dis', 'fea']
dic_path = r'E:/Yue/Entire Data/iemocap_ACMMM_2019/dic_iemocap.txt'
label_path = r'E:/Yue/Entire Data/iemocap_ACMMM_2019/openSmile_label.txt'
text_path = r'E:/Yue/Entire Data/iemocap_ACMMM_2019/transcription.txt'
audio_path = r'E:/Yue/Entire Data/iemocap_ACMMM_2019/openSmile/emo_large.npy'
embed_path = r'E:/Yue/Entire Data/ACL_2018_entire/'
maxlen = 600
numclass = 4
batch_size = 32


def get_label(path):
    f = open(path, 'r')
    statistic = {'ang': 0, 'exc': 0, 'sad': 0, 'fru': 0, 'hap': 0, 'neu': 0, 'oth': 0, 'sur': 0, 'dis': 0, 'fea': 0}
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
        elif line.split()[0] == label_category[6]:
            statistic[label_category[6]] += 1
            res.append(4)
        elif line.split()[0] == label_category[7]:
            statistic[label_category[7]] += 1
            res.append(4)
        elif line.split()[0] == label_category[8]:
            statistic[label_category[8]] += 1
            res.append(4)
        elif line.split()[0] == label_category[9]:
            statistic[label_category[9]] += 1
            res.append(4)
    print(statistic)
    return res

def separate_by_emotion(path, data):
    f = open(path, 'r')
    ang = []
    hap_exc_sur = []
    sad_fea_dis = []
    fru_neu = []
    i = 0
    for line in f:
        if line.split()[0] == label_category[0]:
            ang.append(data[i])
        elif line.split()[0] == label_category[1]:
            hap_exc_sur.append(data[i])
        elif line.split()[0] == label_category[2]:
            sad_fea_dis.append(data[i])
        elif line.split()[0] == label_category[3]:
            fru_neu.append(data[i])
        elif line.split()[0] == label_category[4]:
            hap_exc_sur.append(data[i])
        elif line.split()[0] == label_category[5]:
            fru_neu.append(data[i])
        # elif line.split()[0] == label_category[6]:
        #     oth.append(data[i])
        # elif line.split()[0] == label_category[7]:
        #     hap_exc_sur.append(data[i])
        # elif line.split()[0] == label_category[8]:
        #     sad_fea_dis.append(data[i])
        # elif line.split()[0] == label_category[9]:
        #     sad_fea_dis.append(data[i])
        i += 1
    return ang, hap_exc_sur, sad_fea_dis, fru_neu


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


def get_audio_data(path):
    res = np.load(path)
    return res



def separate_dataset(audio_data, text_data, label):
    train_text_data, train_audio_data, test_text_data, test_audio_data = [], [], [], []
    train_label, test_label = [], []
    ang_text, hap_exc_text, sad_text, fru_neu_text = separate_by_emotion(label_path, text_data)
    ang_audio, hap_exc_audio, sad_audio, fru_neu_audio = separate_by_emotion(label_path, audio_data)
    ang_label, hap_exc_label, sad_label, fru_neu_label = separate_by_emotion(label_path, label)
    ang_i = 0
    while ang_i < len(ang_audio):
        if random.randint(0, 100) < 80:
            train_text_data.append(ang_text[ang_i])
            train_audio_data.append(ang_audio[ang_i])
            train_label.append(ang_label[ang_i])
        else:
            test_text_data.append(ang_text[ang_i])
            test_audio_data.append(ang_audio[ang_i])
            test_label.append(ang_label[ang_i])
        ang_i += 1

    hap_exc_i = 0
    while hap_exc_i < len(hap_exc_audio):
        if random.randint(0, 100) < 80:
            train_text_data.append(hap_exc_text[hap_exc_i])
            train_audio_data.append(hap_exc_audio[hap_exc_i])
            train_label.append(hap_exc_label[hap_exc_i])
        else:
            test_text_data.append(hap_exc_text[hap_exc_i])
            test_audio_data.append(hap_exc_audio[hap_exc_i])
            test_label.append(hap_exc_label[hap_exc_i])
        hap_exc_i += 1

    sad_i = 0
    while sad_i < len(sad_audio):
        if random.randint(0, 100) < 80:
            train_text_data.append(sad_text[sad_i])
            train_audio_data.append(sad_audio[sad_i])
            train_label.append(sad_label[sad_i])

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
    train_audio_data, train_text_data, train_label = shuffle(train_audio_data, train_text_data, train_label)
    test_audio_data, test_text_data, test_label = shuffle(test_audio_data, test_text_data, test_label)
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


def get_data():
    dic = get_dictionary(dic_path)
    embed_matrix = initial_embed(dic, embed_path)
    label = get_label(label_path)
    audio_data = get_audio_data(audio_path)
    text_data = get_text_data(text_path, dic)
    train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label_o = separate_dataset(
        audio_data, text_data, label)
    train_label = to_categorical(train_label, num_classes=numclass)
    train_text_data = sequence.pad_sequences(train_text_data, padding='post', truncating='post', maxlen=maxlen)
    test_label = to_categorical(test_label_o, num_classes=numclass)
    test_text_data = sequence.pad_sequences(test_text_data, padding='post', truncating='post', maxlen=maxlen)
    save_data_audio_opensimle_sentence(train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o)
    return np.array(train_audio_data), train_text_data, train_label, np.array(test_audio_data), test_text_data, test_label, test_label_o, embed_matrix, dic


def get_data1():
    dic = get_dictionary(dic_path)
    embed_matrix = initial_embed(dic, embed_path)
    train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o = load_data_audio_opensmile_sentence()
    return train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic
