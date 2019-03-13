import string
from random import shuffle,randint
from nltk.corpus import stopwords
from augmentation_data import *

index = get_ori_index()

text = get_ori_text()

def get_dictionary(path):
    dic = dict()
    f = open(path, 'r')
    for line in f:
        tmp = []
        for word in line.split():
            tmp.append(word)
        dic[tmp[0]] = int(tmp[1])
    f.close()
    return dic
dict = get_dictionary('dic_iemocap.txt')

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

def get_index(text,dict):
    index = []
    for line in text:
        sub_index = []
        for word in line:
            if word in dict:
                sub_index.append(dict[word])
        index.append(sub_index)
    return index

def get_text(index,dict):
    text = []
    for line in index:
        sub_text = []
        for word in line:
            key = get_keys(dict,word)[0]
            sub_text.append(key)
        text.append(sub_text)
    return text

def shuffle_text(index):
    for line in index:
        shuffle(line)

def drop(index,dict,p):
    words = stopwords.words('english')
    text_ = get_text(index, dict)
    for line in text_:
        j = 0
        for word in line:
            if word in words:
                if randint(0,100)< p*100:
                    line[j] = ''
            j+=1
    index_=get_index(text_,dict)
    return index_



