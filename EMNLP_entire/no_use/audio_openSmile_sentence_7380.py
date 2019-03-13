from __future__ import print_function
from no_use.DataLoader_openSmile_7380 import data_generator, get_data
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import numpy as np
from matplotlib import pyplot as plt

batch_size = 64
audio_path = r'E:/Yue/Entire Data/iemocap_ACMMM_2018/Processed_data/norm_sentence_feature.npy'


def save_list(path, data):
    file = open(path, 'w')
    file.write(str(data))
    file.close()


# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

# word-level feature extraction
word_input = Input(shape=(384, ))
audio_input = Dense(256)(word_input)
audio_input = Activation('relu')(audio_input)
audio_input = Dropout(0.1)(audio_input)
audio_input = Dense(128)(audio_input)
audio_input = Activation('relu')(audio_input)
audio_input = Dropout(0.1)(audio_input)
audio_input = Dense(64)(audio_input)
audio_input = Activation('relu')(audio_input)
audio_input = Dropout(0.1)(audio_input)
audio_input = Dense(32)(audio_input)
audio_input = Activation('relu')(audio_input)
audio_input = Dropout(0.1)(audio_input)
audio_prediction = Dense(4, activation='softmax')(audio_input)

audio_model = Model(inputs=word_input, outputs=audio_prediction)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # lr=0.001/0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08
rmsprop = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.00001, momentum=0.0, decay=0.0, nesterov=False)

# retrain
# model_frame.load_weights(r'E:\Yue\Code\ACL_entire\audio_7380\model_frame_4_class.h5')
# audio_model.load_weights(r'E:\Yue\Code\ACL_entire\audio_7380\audio_model_4_class.h5')

audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
audio_model.summary()

audio_acc = 0
audio_loss = 0
loss_train = []
acc_train = []
loss_test = []
acc_test = []
size = 100
epoch = np.linspace(1, size, size)

# retrain

# audio_loss, audio_acc = audio_model.evaluate_generator(
#      data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
#      steps=len(test_audio_data) / batch_size)
# print(audio_loss, audio_acc)

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
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_openSmile\audio_model_4_class.h5')
        audio_acc = acc_a
        audio_loss = loss_a
print(audio_loss, audio_acc)

plt.figure()
plt.plot(epoch, loss_train, label='loss_train')
plt.plot(epoch, acc_train, label='acc_train')
plt.plot(epoch, loss_test, label='loss_test')
plt.plot(epoch, acc_test, label='acc_test')
plt.xlabel("epoch")
plt.ylabel("audio train/test loss and acc 3.3(7380/IEMOCAP_openSmile)")
plt.legend()
plt.show()
save_list(r'E:\Yue\Code\ACL_entire\audio_openSmile\acc_test.txt', acc_test)
save_list(r'E:\Yue\Code\ACL_entire\audio_openSmile\loss_test.txt', loss_test)
save_list(r'E:\Yue\Code\ACL_entire\audio_openSmile\acc_train.txt', acc_train)
save_list(r'E:\Yue\Code\ACL_entire\audio_openSmile\loss_train.txt', loss_train)
