import scipy.io as sio
import numpy as np
import os
import math

path_to_mat = '.\\MFSC_Nor'
path_to_rule = '.\\word-alignment'
output_path = '.\\Word_Mat_Nor'


def get_rules(rule):
    # get start & end frame of each word --- tuple
    # save start & end frame of all words as a list
    rules = []
    for line in rule:
        start, end, word = line.split('\t')
        start = float(start) * 100
        end = float(end) * 100
        start = int(math.floor(start))
        end = int(math.ceil(end))
        rules.append((start, end))
    return rules


def get_fragments(array, rules, num, filename):
    sum = 0
    for i,rule in enumerate(rules):
        i = str(i)
        start, end = rule
        new_array = array[:,start:end+1]
        if new_array.shape[0]==0 or new_array.shape[1]==0:
            zeroLog = open('zeroLog.txt', 'a')
            zeroLog.write(filename + ' Potential empty mat error\n')
            zeroLog.close()
        else:
            print('Shape after converting: ', new_array.shape)
            sum += (new_array.shape)[1]
            sio.savemat(output_path+'\\'+str(num)+'\\'+i, mdict={'z1':new_array})
    print(sum)


for filename in os.listdir(path_to_mat):
    # load file and gather information
    num, suffix = filename.split('.')
    mat_file = path_to_mat + '\\' + filename
    print ('Working on ' + filename)
    try:
        mat = sio.loadmat(mat_file)
    except TypeError:
        print("Error load mat file: ", filename)
        logfile = open('errLog.txt', 'a')
        logfile.write('Error load ' + filename + '\n')
        logfile.close()
    array = None
    for k in mat.keys():
        if not k.startswith('__'):
            array = mat.get(k)		#ndarray (64, n)
            shape = array.shape
            print('Shape before converting: ', shape)
    rule = open(path_to_rule+'\\'+num+'.txt', 'r')
    rules = get_rules(rule)

    # cut sentence level mat file into word level mat file
    os.mkdir(output_path+'\\'+str(num))
    get_fragments(array, rules, num, filename)