from scipy.io import wavfile
import numpy as np
from sklearn import preprocessing


def normalize(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)


def get_segment(start_time, end_time, data):
    array = np.zeros(2 * 16000,dtype='float64')
    if end_time - start_time > 2:
        # 如果超过了2， 取前2*16000个点
        array = data[int(round(start_time * 16000)): int(round((start_time + 2) * 16000))]
    else:
        # padding到2*16000
        array[0: int(round((end_time - start_time) * 16000))] \
            = data[int(round(start_time * 16000)): int(round(end_time * 16000))]
    return array
