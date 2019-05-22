from keras.layers import Input, Embedding, TimeDistributed
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras import backend as K
from keras import regularizers as rl
from keras import Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras_preprocessing import sequence
from keras.layers import Multiply
from keras.layers import multiply
from tensorflow.python.keras import backend
from Document_level_analysis.self_attention_fusion import Position_Embedding, Attention
from Document_level_analysis.session.codes.visualization.attention import FusionAttention, Light_FusionAttention
from Document_level_analysis.session.codes.visualization.attention import Light_FusionAttention_1, Light_FusionAttention_2
from Document_level_analysis.session.codes.self_attention import SelfAttention
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import numpy as np
from Document_level_analysis.session.codes.get_FLOPs import get_flops
import tensorflow as tf
from word2vec import get_dictionary, initial_embed
from keras import backend as K
from keras import backend

data_path = r'E:\Yue\Code\ACL_entire\Document_level_analysis\session\codes\visualization\end_2end_data\\'
#data_path =r'E:\Yue\Code\ACL_entire\Document_level_analysis\session\codes\visualization\data_new\\'
save_path = r'E:\Yue\Code\ACL_entire\Document_level_analysis\session\codes\visualization\model\\'

n_head = 10
d_k = 20
activation = 'tanh'
dense_size = 64
dropout = 0.25
batch_size = 16   # 58

acc = 0
mae = 100
loss = 1000
loss_train = []
mae_train = []
loss_test = []
mae_test = []
size = 100
epoch = np.linspace(1, size, size)


def get_last_sentense_index(start_index, end_index):
    label = open(r'C:\Users\Han\Desktop\Iemocap\label_multi.txt', 'r')
    trans = open(r'C:\Users\Han\Desktop\Iemocap\transcription_new.txt', 'r')
    label_list = []
    trans_list = []
    for l in label:
        label_list.append(l.strip())
    for t in trans:
        trans_list.append(t.strip())
    test_index = np.load(
        "E:\\Yue\\Code\\ACL_entire\\Document_level_analysis\\session\\pretrain_fininsh\\npy""\\test_sentence_index.npy")
    index1 = []  # index in whole dataset
    index2 = []  # second index to get last sentense in 2017
    index3 = []  # third index to get attention of last sentense
    for case in test_index:
        index1.append(case[-1])
    for case in test_index:
        index2.append(len(case) - 1)
    sum = 0
    for case in test_index:
        sum += len(case) - 1
        index3.append(sum)

    score_array_text = np.load(save_path + 'test_text_att.npy')
    attention_text = []
    for index in range(len(index2)):
        temp = score_array_text[index3[index]]
        multi_head = []
        for head in temp:
            # multi_head.append(head[index2[index]])
            # multi_head.append(head[0:25])
            multi_head.append(head[start_index:end_index])
        attention_text.append(multi_head)

    score_array_audio = np.load(save_path + 'test_audio_att.npy')
    attention_audio = []
    for index in range(len(index2)):
        temp = score_array_audio[index3[index]]
        multi_head = []
        for head in temp:
            # multi_head.append(head[index2[index]])
            # multi_head.append(head[0:25])
            multi_head.append(head[start_index:end_index])
        attention_audio.append(multi_head)

    multi_label = []
    transcriptions = []
    # for index in index1:
    # multi_label.append(label_list[index-25:index])
    # transcriptions.append(trans_list[index-25:index])
    # multi_label.append(label_list[index-(end_index-start_index):index])
    # transcriptions.append(trans_list[index-(end_index-start_index):index])
    for i in range(len(index1)):
        temp = label_list[index1[i] - index2[i] + start_index:index1[i] - index2[i] + end_index]
        sub = []
        for j in temp:
            sub.append(j.split())
        multi_label.append(sub)
        transcriptions.append(trans_list[index1[i] - index2[i] + start_index:index1[i] - index2[i] + end_index])
    return np.array(attention_text), np.array(attention_audio), np.array(transcriptions), np.array(multi_label)


def get_data(path):
    path1 = r'audio\IS_09(384)\\'
    print('loading data...')
    test_a = np.load(path+path1+'test_audio.npy')
    test_t = np.load(path + 'test_text.npy')
    test_l = np.load(path + 'test_label.npy')
    test_a_last = np.load(path+path1+'test_audio_last.npy')
    test_t_last = np.load(path + 'test_text_last.npy')
    print('finish loading data...')

    return test_a, test_t,test_l, test_a_last, test_t_last


def expand_dimensions(x):
    return K.expand_dims(x)


def remove_dimensions(x):
    return K.squeeze(x, axis=1)


def analyze_data(label, predict):
    r_0 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_1 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_2 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_3 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_4 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_5 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_6 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_7 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_8 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    r_9 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    i = 0
    labels = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    while i < len(label):  # 4
        num_l = compute_same_label(label[i])
        num_p = compute_same_label(predict[i])
        if num_l == 0 and num_p == 0:
            if np.argmax(label[i]) == 0:
                labels[str(0)] += 1
                r_0[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i]) == 1:
                labels[str(1)] += 1
                r_1[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i]) == 2:
                labels[str(2)] += 1
                r_2[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i]) == 3:
                labels[str(3)] += 1
                r_3[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i]) == 4:
                labels[str(4)] += 1
                r_4[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i]) == 5:
                labels[str(5)] += 1
                r_5[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i]) == 6:
                labels[str(6)] += 1
                r_6[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i]) == 7:
                labels[str(7)] += 1
                r_7[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i]) == 8:
                labels[str(8)] += 1
                r_8[str(np.argmax(predict[i]))] += 1
            elif np.argmax(label[i]) == 9:
                labels[str(9)] += 1
                r_9[str(np.argmax(predict[i]))] += 1
        i += 1
    print('final result: ')
    print(labels)
    print("0", r_0)
    print("1", r_1)
    print("2", r_2)
    print("3", r_3)
    print("4", r_4)
    print("5", r_5)
    print("6", r_6)
    print("7", r_7)
    print("8", r_8)
    print("9", r_9)


def weight_dot(inputs):
    x = inputs[0]
    y = inputs[1]
    return x * y


def compute_same_label(label):
    flag = 0
    count = 0
    for val in label:
        if val > flag:
            flag = val
            count = 0
        elif val == flag:
            count += 1
    return count


def compute_acc(predict, label):
    accuracy = 0
    count = 0
    # print(predict.shape, label.shape)
    for l in range(len(label)):
        num_l = compute_same_label(label[l])
        num_p = compute_same_label(predict[l])
        if num_l == 0 and num_p == 0:
            if np.argmax(predict[l]) == np.argmax(label[l]):
                accuracy += 1
            count += 1
    # print(count)
    return accuracy / count


def compute_acc_9class(predict, label, without):
    accuracy = 0
    count = 0
    # print(predict.shape, label.shape)
    for l in range(len(label)):
        num_l = compute_same_label(label[l])
        num_p = compute_same_label(predict[l])
        if num_l == 0 and num_p == 0 and np.argmax(label[l]) != without:
            if np.argmax(predict[l]) == np.argmax(label[l]):
                accuracy += 1
            count += 1
    # print(count)
    return accuracy / count


def remove_dimensions(x):
    return K.squeeze(x, axis=1)


# Load the data
test_audio, test_text, test_label, test_audio_last, test_text_last = get_data(data_path)
embed_path = r'E:/Yue/Entire Data/ACL_2018_entire/'
dic_path = r'E:/Yue/Entire Data/iemocap_ACMMM_2018/dic_iemocap.txt'
dic = get_dictionary(dic_path)
embed_matrix = initial_embed(dic, embed_path)

text_sentence_input = Input(shape=(50, 200))
text_sentence_input1 = Position_Embedding()(text_sentence_input)
text_sentence_att = Attention(n_head=10, d_k=20)([text_sentence_input1, text_sentence_input1, text_sentence_input1])
text_sentence_att = BatchNormalization()(text_sentence_att)
text_sentence_att = Activation('tanh')(text_sentence_att)
text_att_gap = GlobalAveragePooling1D()(text_sentence_att)
text_sentence_model = Model(text_sentence_input, text_att_gap)


# Model Structure
audio_input = Input(shape=(167, 384))
text_input = Input(shape=(167, 50))
audio_input_last = Input(shape=(1, 384))
text_input_last = Input(shape=(1, 50))

em_text = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)  # (167,50,200)
em_text_last = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input_last)
text_document_rep = TimeDistributed(text_sentence_model)(em_text)  # (167,200)
text_document_rep_last = TimeDistributed(text_sentence_model)(em_text_last)  # (1,200)


att_a, att_t = Light_FusionAttention_1(n_head=n_head, d_k=d_k)(
    [audio_input, text_document_rep, audio_input_last, text_document_rep_last])
att_a = BatchNormalization()(att_a)
att_a = Activation(activation)(att_a)
att_t = BatchNormalization()(att_t)
att_t = Activation(activation)(att_t)
att_a = Dropout(dropout)(att_a)
att_t = Dropout(dropout)(att_t)

# att_a, att_t = Light_FusionAttention_1(n_head=n_head, d_k=d_k)(
#     [audio_input, text_document_rep, att_a, att_t])
# att_a = BatchNormalization()(att_a)
# att_a = Activation(activation)(att_a)
# att_t = BatchNormalization()(att_t)
# att_t = Activation(activation)(att_t)

fusion_att_a, fusion_att_t = Light_FusionAttention_2(n_head=n_head, d_k=d_k)(
    [audio_input, text_document_rep, att_a, att_t])
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation(activation)(fusion_att_a)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation(activation)(fusion_att_t)
fusion_att_a = Dropout(dropout)(fusion_att_a)
fusion_att_t = Dropout(dropout)(fusion_att_t)

att_a = Dense(200)(att_a)
att_a = BatchNormalization()(att_a)
att_a = Activation('tanh')(att_a)

# fusion_att_a, fusion_att_t = Light_FusionAttention_2(n_head=n_head, d_k=d_k)(
#     [audio_input, text_document_rep, fusion_att_a, fusion_att_t])
# fusion_att_a = BatchNormalization()(fusion_att_a)
# fusion_att_a = Activation(activation)(fusion_att_a)
# fusion_att_t = BatchNormalization()(fusion_att_t)
# fusion_att_t = Activation(activation)(fusion_att_t)

concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
fusion_att_a = concat([fusion_att_a, att_a])
fusion_att_t = concat([fusion_att_t, att_t])
fusion_att_concat = concat([fusion_att_a, fusion_att_t])
print(fusion_att_concat.shape)
# fusion_att_concat = Lambda(remove_dimensions)(fusion_att_concat)


# fusion_att_concat = Dense(256)(fusion_att_concat)
# fusion_att_concat = BatchNormalization()(fusion_att_concat)
# fusion_att_concat = Activation('tanh')(fusion_att_concat)

fusion_att_concat = SelfAttention(n_head=n_head, d_k=d_k)([fusion_att_concat, fusion_att_concat, fusion_att_concat])
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('tanh')(fusion_att_concat)
fusion_att_concat = GlobalAveragePooling1D()(fusion_att_concat)
fusion_att_concat = Dropout(dropout)(fusion_att_concat)

fusion_att_concat = Dense(64)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('tanh')(fusion_att_concat)

# fusion_att_concat = Dense(16)(fusion_att_concat)
# fusion_att_concat = BatchNormalization()(fusion_att_concat)
# fusion_att_concat = Activation('tanh')(fusion_att_concat)

prediction = Dense(10)(fusion_att_concat)
model = Model(inputs=[audio_input, text_input, audio_input_last, text_input_last], outputs=prediction)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
model.summary()
print(get_flops(model))


model.load_weights(save_path + 'fusion_context_5_3_384_update.h5')
output_test = model.predict([test_audio, test_text, test_audio_last, test_text_last], batch_size=1)
acc = compute_acc_9class(output_test, test_label, 6)
analyze_data(test_label, output_test)
print(acc)
