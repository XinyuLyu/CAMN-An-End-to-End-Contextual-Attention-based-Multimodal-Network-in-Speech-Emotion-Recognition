from __future__ import print_function
from keras import backend
from self_attention_hybrid import Position_Embedding,Attention
from DataLoader_7380_5class import *  # process_train_data
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, concatenate, \
    GlobalAveragePooling1D, GlobalMaxPooling1D, TimeDistributed, Lambda
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt

audio_path = r'E:\\Yue\\Entire Data\\iemocap_ACMMM_2018\\Processed_data\\IEMOCAP_Mat_Nor_Align_std_500_40\\'
def save_list(path,data):
    file = open(path, 'w')
    file.write(str(data))
    file.close()
# loading data
print('Loading data...')
train_audio_data, train_text_data, train_label, test_audio_data, test_text_data, test_label, test_label_o, embed_matrix, dic = get_data()

# Audio branch
audio_input = Input(shape=(500, 40))
audio_att = Attention(4, 10)([audio_input, audio_input, audio_input])
audio_att = Attention(4, 10)([audio_att, audio_att, audio_att])
audio_att_gap = GlobalMaxPooling1D()(audio_att)
model_frame = Model(audio_input, audio_att_gap)

word_input = Input(shape=(50, 500, 40))
audio_input = TimeDistributed(model_frame)(word_input)
word_att = Attention(4, 10)([audio_input, audio_input, audio_input])
word_att = Attention(4, 10)([word_att, word_att, word_att])
word_att_gap = GlobalMaxPooling1D()(word_att)
audio_prediction = Dense(5, activation='softmax')(word_att_gap)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=word_input, outputs=[word_att])
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
'''
audio_input = Input(shape=(500, 40))
audio_att = Attention(4, 10)([audio_input, audio_input, audio_input])
audio_att = Dropout(0.05)(audio_att)
audio_att1 = Attention(4, 10)([audio_att, audio_att, audio_att])
audio_att2 = Dropout(0.05)(audio_att1)
audio_att_gap = GlobalMaxPooling1D()(audio_att2)
#audio_att_gap = Lambda(lambda x: backend.sum(x, axis=1))(audio_att1)
model_frame = Model(audio_input, audio_att_gap)

word_input = Input(shape=(50, 500, 40))
word_input1 = TimeDistributed(model_frame)(word_input)
word_att = Attention(4, 10)([word_input1, word_input1, word_input1])
word_att = Dropout(0.05)(word_att)
word_att1 = Attention(4, 10)([word_att, word_att, word_att])
word_att1 = Dropout(0.05)(word_att1)
word_att2 = Attention(4, 10)([word_att1, word_att1, word_att1])
word_att2 = Dropout(0.05)(word_att2)
word_att_gap = GlobalMaxPooling1D()(word_att2)
#word_att_gap = Lambda(lambda x: backend.sum(x, axis=1))(word_att1)
audio_prediction = Dense(5, activation='softmax')(word_att_gap)
audio_model = Model(inputs=word_input, outputs=audio_prediction)
inter_audio_model = Model(inputs=word_input, outputs=word_att1)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
audio_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
'''
# Text Branch(adam)
text_input = Input(shape=(50,))
em_text = Embedding(len(dic) + 1, 200, weights=[embed_matrix], trainable=True)(text_input)
em_text = Position_Embedding()(em_text)
text_att = Attention(4, 10)([em_text, em_text, em_text])
text_att = Dropout(0.05)(text_att)
text_att1 = Attention(4, 10)([text_att, text_att, text_att])
text_att1 = Dropout(0.05)(text_att1)
text_att_gap = GlobalMaxPooling1D()(text_att1)
text_prediction = Dense(5, activation='softmax')(text_att_gap)
text_model = Model(inputs=text_input, outputs=text_prediction)
inter_text_model = Model(inputs=text_input, outputs=text_att1)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
text_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fusion Model
audio_f_input = Input(shape=(50, 40))  # 50，    #98,200      #98,
text_f_input = Input(shape=(50, 200))  # ，64 #98,200      #98,513,64
merge = concatenate([text_f_input, audio_f_input], name='merge')
merge = Dropout(0.05)(merge)
merge_gmp = GlobalMaxPooling1D()(merge)
d_1 = Dense(256)(merge_gmp)
activation1 = Activation('tanh')(d_1)
d_drop1 = Dropout(0.05)(activation1)
d_2 = Dense(128)(d_drop1)
activation2 = Activation('tanh')(d_2)
d_drop2 = Dropout(0.05)(activation2)

f_prediction = Dense(5, activation='softmax')(d_drop2)
final_model = Model(inputs=[text_f_input, audio_f_input], outputs=f_prediction)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

size_t = 40
loss_text_train = []
acc_text_train = []
loss_text_test = []
acc_text_test = []
epoch_text = np.linspace(1,size_t,size_t)
text_acc = 0
text_loss = 100
train_text_inter = None
test_text_inter = None
for i in range(size_t):
    print('text branch, epoch: ', str(i))
    history_t = text_model.fit(train_text_data, train_label, batch_size=batch_size, epochs=1, verbose=1)
    loss_text_train.append(history_t.history['loss'])
    acc_text_train.append(history_t.history['acc'])
    loss_t, acc_t = text_model.evaluate(test_text_data, test_label, batch_size=batch_size, verbose=0)
    loss_text_test.append(loss_t)
    acc_text_test.append(acc_t)
    print('epoch: ', str(i))
    print('loss_t', loss_t, ' ', 'acc_t', acc_t)
    if loss_t <= text_loss:
        text_loss = loss_t
        text_acc = acc_t
        train_text_inter = inter_text_model.predict(train_text_data, batch_size=batch_size)
        test_text_inter = inter_text_model.predict(test_text_data, batch_size=batch_size)
        text_model.save_weights(r'E:\Yue\Code\ACL_entire\validation\\text_model_5_class.h5')
        inter_text_model.save_weights(r'E:\Yue\Code\ACL_entire\validation\\inter_text_model_5_class.h5')

size_a = 50
loss_audio_train = []
acc_audio_train = []
loss_audio_test = []
acc_audio_test = []
epoch_audio = np.linspace(1,size_a,size_a)
train_audio_inter = None
test_audio_inter = None
audio_acc = 0
audio_loss = 100
for i in range(size_a):
    print('audio branch, epoch: ', str(i))
    history_a=audio_model.fit_generator(data_generator(audio_path, train_audio_data, train_label, len(train_audio_data)),
                              steps_per_epoch=len(train_audio_data) / batch_size, epochs=1, verbose=1)
    loss_audio_train.append(history_a.history['loss'])
    acc_audio_train.append(history_a.history['acc'])
    loss_a, acc_a = audio_model.evaluate_generator(
        data_generator(audio_path, test_audio_data, test_label, len(test_audio_data)),
        steps=len(test_audio_data) / batch_size)
    loss_audio_test.append(loss_a)
    acc_audio_test.append(acc_a)
    print('epoch: ', str(i))
    print('loss_a', loss_a, ' ', 'acc_a', acc_a)
    if loss_a <= audio_loss:
        audio_model.save_weights(r'E:\Yue\Code\ACL_entire\validation\audio_model_5_class.h5')
        inter_audio_model.save_weights(r'E:\Yue\Code\ACL_entire\validation\inter_audio_model_5_class.h5')
        model_frame.save_weights(r'E:\Yue\Code\ACL_entire\validation\frame_model_5_class.h5')
        audio_loss = loss_a
        audio_acc = acc_a
        train_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, train_audio_data, train_label, len(train_audio_data)),
            steps=len(train_audio_data))
        test_audio_inter = inter_audio_model.predict_generator(
            data_generator_output(audio_path, test_audio_data, test_label, len(test_audio_data)),
            steps=len(test_audio_data))

size_f = 200
loss_fusion_train = []
acc_fusion_train = []
loss_fusion_test = []
acc_fusion_test = []
epoch_fusion = np.linspace(1,size_f,size_f)
final_acc = 0
final_loss = 100
result = None
for i in range(size_f):
    print('fusion branch, epoch: ', str(i))
    history_f = final_model.fit([train_text_inter, train_audio_inter], train_label, batch_size=batch_size, epochs=1)
    loss_fusion_train.append(history_f.history['loss'])
    acc_fusion_train.append(history_f.history['acc'])
    loss_f, acc_f = final_model.evaluate([test_text_inter, test_audio_inter], test_label, batch_size=batch_size,
                                         verbose=0)
    loss_fusion_test.append(loss_f)
    acc_fusion_test.append(acc_f)
    print('epoch: ', str(i))
    print('loss_f', loss_f, ' ', 'acc_f', acc_f)
    if loss_f <= final_loss:
        final_model.save_weights(r'E:\Yue\Code\ACL_entire\validation\fusion_model_5_class.h5')
        final_loss = loss_f
        final_acc = acc_f
        result = final_model.predict([test_text_inter, test_audio_inter], batch_size=batch_size)
        result = np.argmax(result, axis=1)

r_0, r_1, r_2, r_3, r_4 = analyze_data(test_label_o, result)
print('final result: ')
print('text loss: ', text_loss, ' audio loss: ', audio_loss, ' final loss: ', final_loss)
print('text acc: ', text_acc, ' audio acc: ', audio_acc, ' final acc: ', final_acc)
print("0", r_0)
print("1", r_1)
print("2", r_2)
print("3", r_3)
print("4", r_4)

plt.figure(1)
plt.title('audio')
plt.plot(epoch_audio, loss_audio_train, label='train_loss')
plt.plot(epoch_audio, acc_audio_train, label='train_acc')
plt.plot(epoch_audio, loss_audio_test, label='test_loss')
plt.plot(epoch_audio, acc_audio_test, label='test_acc')
plt.legend()
plt.figure(2)

plt.title('text')
plt.plot(epoch_text, loss_text_train, label='train_loss')
plt.plot(epoch_text, acc_text_train, label='train_acc')
plt.plot(epoch_text, loss_text_test, label='test_loss')
plt.plot(epoch_text, acc_text_test, label='test_acc')
plt.legend()

plt.figure(3)
plt.title('fusion')
plt.plot(epoch_fusion, loss_fusion_train, label='train_loss')
plt.plot(epoch_fusion, acc_fusion_train, label='train_acc')
plt.plot(epoch_fusion, loss_fusion_test, label='test_loss')
plt.plot(epoch_fusion, acc_fusion_test, label='test_acc')
plt.legend()
plt.show()
