from __future__ import print_function
from self_attention_hybrid import Attention
from DataLoader_audio_openSmile_sentence_level import get_data1, get_data
from keras.models import Model
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Input
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

batch_size = 32
dropout = 0.25


def save_list(path, data):
    file = open(path, 'w')
    file.write(str(data))
    file.close()


def reshape(x):
    backend.permute_dimensions(x, (0, 2, 1))
    return x


def expand_dimensions(x):
    return backend.expand_dims(x)


# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

# sentence-level feature extraction
audio_input = Input(shape=(6553,))

audio_att_gap = Dense(2048)(audio_input)
audio_att_gap = Activation('relu')(audio_att_gap)
# audio_att_gap = BatchNormalization()(audio_att_gap)
audio_att_gap = Dropout(dropout)(audio_att_gap)

audio_att_gap = Dense(1024)(audio_att_gap)
audio_att_gap = Activation('relu')(audio_att_gap)
# audio_att_gap = BatchNormalization()(audio_att_gap)
audio_att_gap = Dropout(dropout)(audio_att_gap)

audio_att_gap = Dense(512)(audio_att_gap)
audio_att_gap = Activation('relu')(audio_att_gap)
# audio_att_gap = BatchNormalization()(audio_att_gap)
audio_att_gap = Dropout(dropout)(audio_att_gap)

audio_att_gap = Dense(128)(audio_att_gap)
audio_att_gap = Activation('relu')(audio_att_gap)
# audio_att_gap = BatchNormalization()(audio_att_gap)
audio_att_gap = Dropout(dropout)(audio_att_gap)

audio_att_gap = Dense(32)(audio_att_gap)
audio_att_gap = Activation('relu')(audio_att_gap)
# audio_att_gap = BatchNormalization()(audio_att_gap)
audio_att_gap = Dropout(dropout)(audio_att_gap)

audio_prediction = Dense(4, activation='softmax')(audio_att_gap)

audio_model = Model(inputs=audio_input, outputs=audio_prediction)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08
rmsprop = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

# retrain
# audio_model.load_weights(r'E:\Yue\Code\ACL_entire\audio_opensmile_sentence\audio_model_4_class.h5')

audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
audio_model.summary()

audio_acc = 0
audio_loss = 100
loss_train = []
acc_train = []
loss_test = []
acc_test = []
size = 40
epoch = np.linspace(1, size, size)

# retrain
'''
audio_loss, audio_acc = audio_model.evaluate_generator(
     data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
     steps=len(test_audio_data) / batch_size)
print(audio_loss, audio_acc)
'''

for i in range(size):
    print('audio branch, epoch: ', str(i))
    history = audio_model.fit(train_audio_data, train_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_train.append(history.history['loss'])
    acc_train.append(history.history['acc'])
    loss_a, acc_a = audio_model.evaluate(test_audio_data, test_label, batch_size=batch_size, verbose=0)
    acc_test.append(acc_a)
    loss_test.append(loss_a)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if loss_a <= audio_loss:
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\audio_opensmile_sentence\audio_model_4_class.h5')
        audio_acc = acc_a
        audio_loss = loss_a
print(audio_loss, audio_acc)

plt.figure()
plt.plot(epoch, loss_train, label='loss_train')
plt.plot(epoch, acc_train, label='acc_train')
plt.plot(epoch, loss_test, label='loss_test')
plt.plot(epoch, acc_test, label='acc_test')
plt.xlabel("epoch")
plt.ylabel("audio opensmile loss and acc 3.11(sentence)")
plt.legend()
plt.show()

save_list(r'E:\Yue\Code\ACL_entire\audio_opensmile_sentence\acc_test.txt', acc_test)
save_list(r'E:\Yue\Code\ACL_entire\audio_opensmile_sentence\loss_test.txt', loss_test)
save_list(r'E:\Yue\Code\ACL_entire\audio_opensmile_sentence\acc_train.txt', acc_train)
save_list(r'E:\Yue\Code\ACL_entire\audio_opensmile_sentence\loss_train.txt', loss_train)
