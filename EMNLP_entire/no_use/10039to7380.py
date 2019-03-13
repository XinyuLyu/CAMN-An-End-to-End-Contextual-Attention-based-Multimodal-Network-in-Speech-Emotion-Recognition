
label_category = ['ang', 'exc', 'sad', 'fru', 'hap', 'neu', 'oth', 'sur', 'dis', 'fea']
dic_path = r'E:/Yue/Entire Data/iemocap_ACMMM_2018/dic_iemocap.txt'
label_path = r'E:/Yue/Entire Data/iemocap_ACMMM_2018/label.txt'
text_path = r'E:/Yue/Entire Data/iemocap_ACMMM_2018/transcription.txt'
embed_path = r'E:/Yue/Entire Data/ACL_2018_entire/'
maxlen = 50
numclass = 4
batch_size = 16


def get_label(path):
    f = open(path, 'r')
    res = []
    for line in f:
        if line.split()[0] == label_category[0]:
            res.append(0)
        elif line.split()[0] == label_category[1]:
            res.append(1)
        elif line.split()[0] == label_category[2]:
            res.append(2)
        elif line.split()[0] == label_category[3]:
            res.append(3)
        elif line.split()[0] == label_category[4]:
            res.append(4)
        elif line.split()[0] == label_category[5]:
            res.append(5)
        if line.split()[0] == label_category[6]:
            res.append(6)
        elif line.split()[0] == label_category[7]:
            res.append(7)
        elif line.split()[0] == label_category[8]:
            res.append(8)
        elif line.split()[0] == label_category[9]:
            res.append(9)
    return res

def get_text(path):
    result = []
    fd = open(path, "r")
    for line in fd.readlines():
        result.append(line)
    return result

def check(path1, path2):
    text1 = get_text(path1+'transcription.txt')
    text2 = get_text(path2+'text_output_new.txt')
    label1 = get_label(path1 + 'label.txt')
    label2 = get_label(path2 + 'label_output_new.txt')

    j=0
    for i in range(len(text1)):
        f = open(path2+'text_output_new.txt','r')
        for line in f:
            if str(text1[i]) == str(line):
                if label1[i] != label2[j]:
                    print(str(text1[i]), ' 7024('+str(i)+'):' + str(label_category[label1[i]]) , ' 10038('+str(j)+'):' + str(
                        label_category[label2[j]]))
                    j+=1
                else:
                    j+=1
                    break
            else:
                j+=1
if __name__ == '__main__':
    path1 = r'E:\Yue\Entire Data\iemocap_ACMMM_2018\\'
    path2 = r'E:\Yue\Entire Data\ACL_2018_entire\\'
    check(path1, path2)

