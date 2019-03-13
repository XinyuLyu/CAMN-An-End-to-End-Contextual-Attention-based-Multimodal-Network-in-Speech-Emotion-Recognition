import numpy as np

label_category = ['ang', 'exc', 'sad', 'fru', 'hap', 'neu', 'oth', 'sur', 'dis', 'fea']
stat = {'ang': 0, 'exc': 0, 'sad': 0, 'fru': 0, 'hap': 0, 'neu': 0, 'oth': 0, 'sur': 0, 'dis': 0, 'fea': 0}
text_a_path = 'E:\\Yue\\Entire Data\\iemocap_ACMMM_2018\\transcription.txt'
text_b_path = 'E:\\Yue\\Entire Data\\ACL_2018_entire\\text_output_new.txt'
label_a_path = 'E:\\Yue\\Entire Data\\iemocap_ACMMM_2018\\label.txt'
label_b_path = 'E:\\Yue\\Entire Data\\ACL_2018_entire\\label_output_new.txt'


def get_label(path):
    f = open(path, 'r')
    res = []
    for line in f:
        res.append(line.split()[0])
    f.close()
    return res


def get_text(path):
    f = open(path, 'r')
    res = []
    for line in f:
        res.append(line)
    f.close()
    return res


def comparison(text_a, text_b, label_a, label_b):
    for i in range(len(label_a)):
        if label_a[i] in label_category[:6]:
            if text_a[i] in text_b and label_a[i].strip() != label_b[text_b.index(text_a[i])].strip():
                # print('--------------------------------')
                # print(label_a[i].strip())
                # print(label_b[text_b.index(text_a[i])].strip())
                # # print(text_a[i])
                # # print(text_b[text_b.index(text_a[i])])
                # print(label_a[i] != label_b[text_b.index(text_a[i])])
                # print('--------------------------------')

                stat[label_a[i]] += 1
            elif text_a[i] not in text_b:
                stat[label_a[i]] += 1


def evaluate_occurence(text_a, text_b):
    text_a_np = np.array(text_a)
    text_b_np = np.array(text_b)
    res = []
    for i in range(len(text_a)):
        if np.where(text_b_np == text_a[i])[0].shape[0] > 1:
            res.append(text_a[i])
    print(res)
    print(len(res))


text_a = get_text(text_a_path)
text_b = get_text(text_b_path)
label_a = get_label(label_a_path)
label_b = get_text(label_b_path)


evaluate_occurence(text_a, text_b)
# comparison(text_a, text_b, label_a, label_b)
#
# print(stat)
