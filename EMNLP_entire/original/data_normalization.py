import numpy as np
import scipy.io as scio
from sklearn import preprocessing
audio_path = r'E:\Yue\Entire Data\iemocap_ACMMM_2019\MFSC_source\IEMOCAP_Mat_64_update\\'
output_path = r'E:\Yue\Entire Data\iemocap_ACMMM_2019\Data_normalization\IEMOCAP_Mat_64_update\\'
num = 10039


def get_mat_data(path):
    res, length = [], []
    i = 0
    while i < num:
        print('step: ', i)
        tmp = scio.loadmat(path+str(i)+".mat")
        tmp = tmp['z1']
        length.append(tmp.shape[1])
        #tmp = tmp.transpose()
        if res == []:
            res = tmp
        else:
            res = np.concatenate((res, tmp), axis=1)
        #res.append(tmp)
        i += 1
    #res = np.array(res)
    return res, length


def output_mat_data(path, res, length):
    i = 0
    while i < num:
        if i == len(length)-1:
            tmp = res
        else:
            tmp, res = np.split(res, [length[i]], axis=1)
        scio.savemat(path+str(i)+".mat", {'z1': tmp})
        i += 1


data, length = get_mat_data(audio_path)
print(len(length))
print(data.shape)
# min_max_scaler = preprocessing.MinMaxScaler()
# data = min_max_scaler.fit_transform(data)
data = preprocessing.scale(data)
output_mat_data(output_path, data, length)
