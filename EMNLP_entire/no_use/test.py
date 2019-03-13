import numpy as np
audio_path = r'E:/Yue/Entire Data/iemocap_ACMMM_2018/Processed_data/norm_sentence_feature.npy'
data = np.load(audio_path)
print(data.shape)
print(data[100])


