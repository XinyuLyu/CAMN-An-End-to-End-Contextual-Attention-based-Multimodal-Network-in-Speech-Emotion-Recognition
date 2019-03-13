from __future__ import print_function
from self_attention_hybrid import Attention, LayerNormalization, Position_Embedding
from DataLoader_audio_7204 import get_data
from DataLoader_audio_7204 import analyze_data
from DataLoader_audio_7204 import data_generator
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
audio_path = r'E:\\Yue\\Entire Data\\ACL_2018_entire\\Word_Mat_New_1\\'
def save_list(path,data):
    file = open(path, 'w')
    file.write(str(data))
    file.close()

# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

# Frame-level feature extraction
audio_input = Input(shape=(513, 64))
audio_input1 = Position_Embedding()(audio_input)
# LSTM test
# audio_att_gap = Bidirectional(LSTM(32, return_sequences=False, recurrent_dropout=0.25))(audio_input)
# audio_att_gap = Bidirectional(LSTM(32, recurrent_dropout=0.25))(audio_att_gap)
#self-attention frame domain test
'''
audio_att1 = Attention(n_head=4, d_k=10)([audio_input1, audio_input1, audio_input1])
audio_att2 = Attention(n_head=4, d_k=10)([audio_att1, audio_att1, audio_att1])
audio_att_gap = GlobalMaxPooling1D()(audio_att2)
'''
# CNN test

cnn_1 = Conv1D(64, 2, padding='valid', strides=1)(audio_input)
cnn_1 = Activation('relu')(cnn_1)
cnn_1 = GlobalMaxPooling1D()(cnn_1)
cnn_1 = Dropout(0.25)(cnn_1)

cnn_2 = Conv1D(64, 3, padding='valid', strides=1)(audio_input)
cnn_2 = Activation('relu')(cnn_2)
cnn_2 = GlobalMaxPooling1D()(cnn_2)
cnn_2 = Dropout(0.25)(cnn_2)

cnn_3 = Conv1D(64, 4, padding='valid', strides=1)(audio_input)
cnn_3 = Activation('relu')(cnn_3)
cnn_3 = GlobalMaxPooling1D()(cnn_3)
cnn_3 = Dropout(0.25)(cnn_3)

cnn_4 = Conv1D(64, 5, padding='valid', strides=1)(audio_input)
cnn_4 = Activation('relu')(cnn_4)
cnn_4 = GlobalMaxPooling1D()(cnn_4)
cnn_4 = Dropout(0.25)(cnn_4)

audio_att_gap = concatenate([cnn_1, cnn_2, cnn_3, cnn_4])
print('audio_att_gap shape: ', audio_att_gap.shape)
audio_att_gap = Dense(64)(audio_att_gap)
audio_att_gap = Activation('relu')(audio_att_gap)
audio_att_gap = Dropout(0.25)(audio_att_gap)

# frame-level model build
model_frame = Model(audio_input, audio_att_gap)
model_frame.summary()

# word-level feature extraction
word_input = Input(shape=(50, 513, 64))
word_input1 = TimeDistributed(model_frame)(word_input)
word_att1 = Attention(n_head=4, d_k=10)([word_input1, word_input1, word_input1])
word_att2 = Attention(n_head=4, d_k=10)([word_att1, word_att1, word_att1])
word_att_gap = GlobalMaxPooling1D()(word_att2)
audio_prediction = Dense(4, activation='softmax')(word_att_gap)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
# inter_audio_model = Model(inputs=word_input, outputs=word_att2)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08
#model_frame.load_weights(r'E:\Yue\Code\ACL_entire\audio_attention\model_frame_4_class.h5')
#audio_model.load_weights(r'E:\Yue\Code\ACL_entire\audio_attention\audio_model_4_class.h5')
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

audio_acc = 0
audio_loss = 0
loss_train = []
acc_train = []
loss_test = []
acc_test = []
size = 50
epoch = np.linspace(1, size, size)

# re train
'''
audio_loss, audio_acc = audio_model.evaluate_generator(
     data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
     steps=len(test_audio_data) / batch_size)
print(audio_loss, audio_acc)
'''
for i in range(size):
    print('audio branch, epoch: ', str(i))
    history = audio_model.fit_generator(data_generator(audio_path,
                                                       train_audio_data,
                                                       train_label,
                                                       len(train_audio_data)),
                                        steps_per_epoch=len(train_audio_data) / batch_size,
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
    if acc_a >= audio_acc:
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_attention\audio_model_4_class.h5')
        model_frame.save_weights(r'E:\Yue\Code\ACL_entire\audio_attention\model_frame_4_class.h5')
        audio_acc = acc_a
        audio_loss = loss_a
print(audio_loss, audio_acc)

plt.figure()
plt.plot(epoch, loss_train, label='loss_train')
plt.plot(epoch, acc_train, label='acc_train')
plt.plot(epoch, loss_test, label='loss_test')
plt.plot(epoch, acc_test, label='acc_test')
plt.xlabel("epoch")
plt.ylabel("audio train/test loss and acc 3.4(attention)")
plt.legend()
plt.show()
