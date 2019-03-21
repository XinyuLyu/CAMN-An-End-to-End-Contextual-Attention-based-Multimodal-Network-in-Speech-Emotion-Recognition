from __future__ import print_function

from sklearn.utils import shuffle

from self_attention_hybrid import Attention, LayerNormalization, Position_Embedding
from DataLoader_audio_7380_sentense import get_data,data_generator
from DataLoader_audio_7204 import analyze_data
from DataLoader_audio_7204 import data_generator_output
from DataLoader_audio_7204 import get_data1
from keras.models import Model
from keras.layers import Dense, Add
from keras.layers import Input
from keras.layers import GlobalAveragePooling1D
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import GlobalMaxPooling1D
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import concatenate
from keras.layers import Lambda
from keras.layers import Activation
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from keras import backend
batch_size = 8
audio_path=r'E:\Yue\Entire Data\iemocap_ACMMM_2018\MFSC_source\IEMOCAP_Mat_64\\'
def save_list(path,data):
    file = open(path, 'w')
    file.write(str(data))
    file.close()

# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()
# Frame-level feature extraction
audio_input = Input(shape=(2250, 64))
audio_input1 = Position_Embedding()(audio_input)
audio_att1 = Attention(n_head=4, d_k=16)([audio_input1, audio_input1, audio_input1])
audio_att2 = Attention(n_head=4, d_k=16)([audio_att1, audio_att1, audio_att1])
audio_att_gap = GlobalMaxPooling1D()(audio_att2)
audio_prediction = Dense(4, activation='softmax')(audio_att_gap)
audio_model = Model(inputs=audio_input, outputs=audio_prediction)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

audio_acc = 0
audio_loss = 100
loss_train = []
acc_train = []
loss_test = []
acc_test = []
size = 65
epoch = np.linspace(1, size, size)

for i in range(size):
    print('audio branch, epoch: ', str(i))
    train_d, train_l = shuffle(train_audio_data, train_label)
    history = audio_model.fit_generator(data_generator(audio_path,
                                                       train_d,
                                                       train_l,
                                                       len(train_d)),
                                        steps_per_epoch=len(train_d) / batch_size,
                                        epochs=1,
                                        verbose=1)
    loss_train.append(history.history['loss'])
    acc_train.append(history.history['acc'])
    loss_a, acc_a = audio_model.evaluate_generator(
        data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
        steps=len(test_audio_data) / batch_size)
    acc_test.append(acc_a)
    loss_test.append(loss_a)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if acc_a > audio_acc:
        audio_acc = acc_a
        audio_loss = loss_a
print(audio_acc,audio_loss)

plt.figure()
plt.plot(epoch, loss_train, label='loss_train')
plt.plot(epoch, acc_train, label='acc_train')
plt.plot(epoch, loss_test, label='loss_test')
plt.plot(epoch, acc_test, label='acc_test')
plt.xlabel("epoch")
plt.ylabel("audio train/test loss and acc 3.14(attention)")
plt.legend()
plt.show()
