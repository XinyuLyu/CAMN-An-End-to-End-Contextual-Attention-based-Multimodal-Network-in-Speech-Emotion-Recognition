from __future__ import print_function

from sklearn.utils import shuffle

from original.attention_model import AttentionLayer
from DataLoader_audio_sentence_level import data_generator, get_data1, get_data
from keras.models import Model
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import GlobalMaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras import backend
import numpy as np
from matplotlib import pyplot as plt

batch_size = 16
dropout = 0.25
audio_path = r'E:\Yue\Entire Data\iemocap_ACMMM_2019\STFT_source\\'


def save_list(path, data):
    file = open(path, 'w')
    file.write(str(data))
    file.close()


def reshape(x):
    backend.permute_dimensions(x, (0, 2, 1))
    return x


def expand_dimensions(x):
    return backend.expand_dims(x)


def weight_dot(inputs):
    x = inputs[0]
    y = inputs[1]
    return x * y


def compute_acc(predict, label):
    acc = 0
    # print(predict.shape, label.shape)
    for l in range(len(label)):
        if np.argmax(predict[l]) == np.argmax(label[l]):
            acc += 1
    return acc / len(label)


# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

# sentence-level feature extraction
audio_input = Input(shape=(513, 600))


audio_inputs = Lambda(expand_dimensions)(audio_input)
cnn_1 = Conv2D(32, (3, 3), padding='valid', strides=(1, 1))(audio_inputs)
cnn_1 = BatchNormalization()(cnn_1)
cnn_1 = Activation('relu')(cnn_1)
cnn_1 = MaxPooling2D(pool_size=(2, 2))(cnn_1)
# cnn_1 = Dropout(dropout)(cnn_1)

cnn_2 = Conv2D(64, (3, 3), padding='valid', strides=(1, 1))(cnn_1)
cnn_2 = BatchNormalization()(cnn_2)
cnn_2 = Activation('relu')(cnn_2)
cnn_2 = MaxPooling2D(pool_size=(2, 2))(cnn_2)
# cnn_2 = Dropout(dropout)(cnn_2)

cnn_3 = Conv2D(128, (3, 3), padding='valid', strides=(1, 1))(cnn_2)
cnn_3 = BatchNormalization()(cnn_3)
cnn_3 = Activation('relu')(cnn_3)
cnn_3 = MaxPooling2D(pool_size=(2, 2))(cnn_3)
# cnn_3 = Dropout(dropout)(cnn_3)

cnn_4 = Conv2D(256, (3, 3), padding='valid', strides=(1, 1))(cnn_3)
cnn_4 = BatchNormalization()(cnn_4)
cnn_4 = Activation('relu')(cnn_4)
cnn_4 = MaxPooling2D(pool_size=(2, 2))(cnn_4)
# cnn_4 = Dropout(dropout)(cnn_4)

cnn_5 = Conv2D(512, (3, 3), padding='valid', strides=(1, 1))(cnn_4)
cnn_5 = BatchNormalization()(cnn_5)
cnn_5 = Activation('relu')(cnn_5)
cnn_5 = MaxPooling2D(pool_size=(2, 2))(cnn_5)
# cnn_5 = Dropout(dropout)(cnn_5)

cnn_6 = Conv2D(1024, (3, 3), padding='valid', strides=(1, 1))(cnn_5)
cnn_6 = BatchNormalization()(cnn_6)
cnn_6 = Activation('relu')(cnn_6)
cnn_6 = MaxPooling2D(pool_size=(2, 2))(cnn_6)
# cnn_6 = Dropout(dropout)(cnn_6)

audio_att_gap = Flatten()(cnn_6)

audio_att_gap = Dense(2048)(audio_att_gap)
audio_att_gap = BatchNormalization()(audio_att_gap)
audio_att_gap = Activation('relu')(audio_att_gap)
# audio_att_gap = Dropout(dropout)(audio_att_gap)

audio_att_gap = Dense(1024)(audio_att_gap)
audio_att_gap = BatchNormalization()(audio_att_gap)
audio_att_gap = Activation('relu')(audio_att_gap)
# audio_att_gap = Dropout(dropout)(audio_att_gap)

audio_att_gap = Dense(512)(audio_att_gap)
audio_att_gap = BatchNormalization()(audio_att_gap)
audio_att_gap = Activation('relu')(audio_att_gap)
# audio_att_gap = Dropout(dropout)(audio_att_gap)

audio_att_gap = Dense(128)(audio_att_gap)
audio_att_gap = BatchNormalization()(audio_att_gap)
audio_att_gap = Activation('relu')(audio_att_gap)
# audio_att_gap = Dropout(dropout)(audio_att_gap)

# audio_att_gap = Dense(32)(audio_att_gap)
# audio_att_gap = BatchNormalization()(audio_att_gap)
# audio_att_gap = Activation('relu')(audio_att_gap)
# audio_att_gap = Dropout(dropout)(audio_att_gap)


# audio_input1 = Lambda(reshape)(audio_input)
# audio_l1 = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.25, name='LSTM_audio_1'))(audio_input)
# audio_l2 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.25, name='LSTM_audio_2'))(audio_l1)
# word_weight = AttentionLayer()(audio_l1)
# word_weight_exp = Lambda(expand_dimensions)(word_weight)
# word_attention = Lambda(weight_dot)([audio_l1, word_weight_exp])
# word_att = Lambda(lambda x: backend.sum(x, axis=1))(word_attention)
# audio_att_gap = Dropout(0.25)(word_att)

audio_prediction = Dense(10)(audio_att_gap)

audio_model = Model(inputs=audio_input, outputs=audio_prediction)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08
rmsprop = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

# retrain
# audio_model.load_weights(r'E:\Yue\Code\ACL_entire\audio_stft_sentence\audio_model_4_class.h5')

# audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# mean_squared_error, mean_absolute_error
audio_model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['mae'])
audio_model.summary()

audio_acc = 0
audio_loss = 100
loss_train = []
acc_train = []
loss_test = []
acc_test = []
size = 100
epoch = np.linspace(1, size, size)

# retrain
'''
audio_loss, audio_acc = audio_model.evaluate_generator(
     data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
     steps=len(test_audio_data) / batch_size)
print(audio_loss, audio_acc)
'''
# Classification
# for i in range(size):
#     print('audio branch, epoch: ', str(i))
#     history = audio_model.fit_generator(data_generator(audio_path,
#                                                        train_audio_data,
#                                                        train_label,
#                                                        len(train_audio_data)),
#                                         steps_per_epoch=len(train_audio_data) / batch_size,
#                                         epochs=1,
#                                         verbose=1)
#     loss_train.append(history.history['loss'])
#     acc_train.append(history.history['acc'])
#     loss_a, acc_a = audio_model.evaluate_generator(
#         data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
#         steps=len(test_audio_data) / batch_size)
#     acc_test.append(acc_a)
#     loss_test.append(loss_a)
#     print('epoch: ', str(i))
#     print('loss_a', loss_a, ' ', 'acc_a', acc_a)
#     if loss_a <= audio_loss:
#         audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_stft_sentence\audio_model_4_class.h5')
#         audio_acc = acc_a
#         audio_loss = loss_a
# print(audio_loss, audio_acc)


# Regression
for i in range(size):
    print('audio branch, epoch: ', str(i))
    train_d, train_l = shuffle(train_audio_data, train_label)
    history = audio_model.fit_generator(data_generator(audio_path,
                                                       train_audio_data,
                                                       train_label,
                                                       len(train_audio_data)),
                                        steps_per_epoch=len(train_audio_data) / batch_size,
                                        epochs=1,
                                        verbose=1)
    loss_train.append(history.history['loss'])
    loss_a, loss_mae = audio_model.evaluate_generator(data_generator(audio_path,
                                                           test_audio_data,
                                                           test_label,
                                                           len(test_audio_data)),
                                            steps=len(test_audio_data) / batch_size)
    loss_test.append(loss_a)
    print('epoch: ', str(i))
    print('loss_a', loss_a)
    if loss_a <= audio_loss:
        res_test = audio_model.predict_generator(data_generator(audio_path,
                                                                test_audio_data,
                                                                test_label,
                                                                len(test_audio_data)),
                                                 steps=len(test_audio_data) / batch_size)
        acc = compute_acc(res_test, test_label)
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_stft_sentence\reg_audio_model_4_class.h5')
        audio_loss = loss_a
        audio_acc = acc
    print('acc', audio_acc)
print(audio_loss, audio_acc)

plt.figure()
plt.plot(epoch, loss_train, label='loss_train')
# plt.plot(epoch, acc_train, label='acc_train')
plt.plot(epoch, loss_test, label='loss_test')
# plt.plot(epoch, acc_test, label='acc_test')
plt.xlabel("epoch")
plt.ylabel("audio stft reg loss")
plt.legend()
plt.show()
# save_list(r'E:\Yue\Code\ACL_entire\audio_stft_sentence\acc_test.txt', acc_test)
save_list(r'E:\Yue\Code\ACL_entire\audio_stft_sentence\loss_test.txt', loss_test)
# save_list(r'E:\Yue\Code\ACL_entire\audio_stft_sentence\acc_train.txt', acc_train)
save_list(r'E:\Yue\Code\ACL_entire\audio_stft_sentence\loss_train.txt', loss_train)
