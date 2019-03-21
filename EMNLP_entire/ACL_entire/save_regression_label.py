import numpy as np

reg_label_path = r'E:\Yue\Entire Data\iemocap_ACMMM_2019\label_multi.txt'
save_path = r'E:\Yue\Entire Data\iemocap_ACMMM_2019\label_multi.npy'


def get_label(path):
    res = []
    f = open(path, 'r')
    for line in f:
        tmp = []
        for i in range(10):
            tmp.append(float(line.split()[i]))
        res.append(tmp)
    f.close()

    return res


if __name__ == "__main__":
    label = get_label(reg_label_path)
    # print(label[0])
    # print(label[1])
    np.save(save_path, label)
