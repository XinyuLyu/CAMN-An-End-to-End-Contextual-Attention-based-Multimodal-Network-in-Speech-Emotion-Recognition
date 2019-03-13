import scipy.io as scio

path = r'E:/Yue/Entire Data/iemocap_ACMMM_2018/IEMOCAP_Mat_Nor_Align_wav/'


def data_generator(path):
    i = 0
    while i < 10039:
        tmp = scio.loadmat(path + str(i) + ".mat")
        tmp = tmp['data']
        print(tmp.shape)
        # if tmp.shape[-1] != 32000:
        #     print('###########################################', str(i), tmp.shape)
        i += 1


data_generator(path)
