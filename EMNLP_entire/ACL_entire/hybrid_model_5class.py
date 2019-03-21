from __future__ import print_function
from self_attention_hybrid import Position_Embedding
from self_attention_hybrid import Attention
from self_attention_hybrid import FusionAttention
from DataLoader_hybrid_5class import get_data
from DataLoader_hybrid_5class import data_generator
from DataLoader_hybrid_5class import data_generator_output
from DataLoader_hybrid_5class import analyze_data
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Lambda
from keras import backend as K
from keras import backend
from sklearn.utils import shuffle
from keras import regularizers as rl


max_features = 20000
batch_size = 6
epo = 100
filters = 128
flag = 0.60
numclass = 5
audio_path = r'E:\Yue\Entire Data\ACL_2018_entire\Word_Mat_New_1\\'


def reshape(x):
    backend.permute_dimensions(x, (0, 2, 1))
    return x


# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

# frame model
audio_input = Input(shape=(513, 64))
# audio_input_tran = Lambda(reshape)(audio_input)
audio_att = Attention(n_head=4, d_k=16)([audio_input, audio_input, audio_input])
audio_att = Dense(64, kernel_regularizer=rl.l2(0.01))(audio_att)
audio_att = BatchNormalization()(audio_att)
audio_att = Activation('relu')(audio_att)
audio_att = Attention(n_head=4, d_k=16)([audio_att, audio_att, audio_att])
audio_att = Dense(64, kernel_regularizer=rl.l2(0.01))(audio_att)
audio_att = BatchNormalization()(audio_att)
audio_att = Activation('relu')(audio_att)
# audio_att = Lambda(reshape)(audio_att)
audio_att_gap = Lambda(lambda x: backend.sum(x, axis=1))(audio_att)
model_frame = Model(audio_input, audio_att_gap)
model_frame.summary()

# audio frame
word_input = Input(shape=(50, 513, 64))
text_input = Input(shape=(50,))
word_rep = TimeDistributed(model_frame)(word_input)
text_emb = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
# word_rep = Position_Embedding()(word_rep)
text_emb = Position_Embedding()(text_emb)

fusion_att_a, fusion_att_t = FusionAttention(n_head=4, d_k=16)([word_rep, text_emb])
fusion_att_a = Dense(64)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation('relu')(fusion_att_a)
fusion_att_t = Dense(64)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation('relu')(fusion_att_t)

fusion_att_a, fusion_att_t = FusionAttention(n_head=4, d_k=16)([fusion_att_a, fusion_att_t])
fusion_att_a = Dense(64)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation('relu')(fusion_att_a)
fusion_att_t = Dense(64)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation('relu')(fusion_att_t)

fusion_att_a, fusion_att_t = FusionAttention(n_head=4, d_k=16)([fusion_att_a, fusion_att_t])
fusion_att_a = Dense(64)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation('relu')(fusion_att_a)
fusion_att_t = Dense(64)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation('relu')(fusion_att_t)

fusion_att_a, fusion_att_t = FusionAttention(n_head=4, d_k=16)([fusion_att_a, fusion_att_t])
fusion_att_a = Dense(64)(fusion_att_a)
fusion_att_a = BatchNormalization()(fusion_att_a)
fusion_att_a = Activation('relu')(fusion_att_a)
fusion_att_t = Dense(64)(fusion_att_t)
fusion_att_t = BatchNormalization()(fusion_att_t)
fusion_att_t = Activation('relu')(fusion_att_t)

# print(fusion_att_a.shape, fusion_att_t.shape)

concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))
fusion_att_concat = concat([fusion_att_a, fusion_att_t])
fusion_att_concat = Lambda(lambda x: backend.sum(x, axis=1))(fusion_att_concat)
fusion_att_concat = Dense(64)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('relu')(fusion_att_concat)
fusion_att_concat = Dense(32)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('relu')(fusion_att_concat)
fusion_att_concat = Dense(16)(fusion_att_concat)
fusion_att_concat = BatchNormalization()(fusion_att_concat)
fusion_att_concat = Activation('relu')(fusion_att_concat)

fusion_prediction = Dense(5, activation='softmax')(fusion_att_concat)
fusion_model = Model(inputs=[word_input, text_input], outputs=fusion_prediction)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# model_frame.load_weights(r'E:\Yue\Code\ACL_entire\hybrid\\frame_4_class.h5')
# fusion_model.load_weights(r'E:\Yue\Code\ACL_entire\hybrid\\fusion_4_class.h5')
fusion_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fusion_model.summary()
# fusion_loss, fusion_acc = fusion_model.evaluate_generator(data_generator(audio_path, test_audio_data,test_text_data, test_label, len(test_audio_data)),steps=len(test_audio_data) / 4)
# result = fusion_model.predict_generator(data_generator_output(audio_path, test_audio_data, test_text_data, test_label, len(test_audio_data)),steps=len(test_audio_data))
# result = np.argmax(result, axis=1)


# train
fusion_acc = 0
fusion_loss = 100
train_fusion_inter = None
test_fusion_inter = None
loss_train = []
acc_train = []
loss_test = []
acc_test = []
result = []
size = 100
epoch = np.linspace(1, size, size)

for i in range(size):
    print('branch, epoch: ', str(i))
    epo_audio_data, epo_text_data, epo_label = shuffle(train_audio_data, train_text_data, train_label)
    history = fusion_model.fit_generator(data_generator(audio_path,
                                                        epo_audio_data,
                                                        epo_text_data,
                                                        epo_label,
                                                        len(train_audio_data)),
                                         steps_per_epoch=len(train_audio_data) / batch_size,
                                         epochs=1,
                                         verbose=1)
    loss_train.append(history.history['loss'])
    acc_train.append(history.history['acc'])
    loss_f, acc_f = fusion_model.evaluate_generator(data_generator(audio_path,
                                                                   test_audio_data,
                                                                   test_text_data,
                                                                   test_label,
                                                                   len(test_audio_data)),
                                                    steps=len(test_audio_data) / batch_size)
    loss_test.append(loss_f)
    acc_test.append(acc_f)
    print('epoch: ', str(i))
    print('loss_f', loss_f, ' ', 'acc_f', acc_f)

    # fusion_a = fusion_inter_model.predict_generator(
    #         data_generator_output(audio_path, test_audio_data, test_text_data, test_label, len(test_audio_data)),
    #         steps=len(test_audio_data))

    if loss_f <= fusion_loss:
        fusion_acc = acc_f
        fusion_loss = loss_f
        fusion_model.save_weights(r'E:\Yue\Code\ACL_entire\hybrid\\fusion_4_class.h5')
        model_frame.save_weights(r'E:\Yue\Code\ACL_entire\hybrid\\frame_4_class.h5')
        result = fusion_model.predict_generator(data_generator_output(audio_path,
                                                                      test_audio_data,
                                                                      test_text_data,
                                                                      test_label,
                                                                      len(test_audio_data)),
                                                steps=len(test_audio_data))
        result = np.argmax(result, axis=1)


# confusion matrix
r_0, r_1, r_2, r_3, r_4= analyze_data(test_label_o, result)
print('final result: ')
print('fusion acc: ', fusion_acc, 'fusion loss', fusion_loss)
print("0", r_0)
print("1", r_1)
print("2", r_2)
print("3", r_3)
print("4", r_4)


# plot and save plot data
plt.figure()
plt.plot(epoch, loss_train, label='loss_train')
plt.plot(epoch, acc_train, label='acc_train')
plt.plot(epoch, loss_test, label='loss_test')
plt.plot(epoch, acc_test, label='acc_test')
plt.xlabel("epoch")
plt.ylabel("loss and acc for hybrid model")
plt.legend()
plt.show()
