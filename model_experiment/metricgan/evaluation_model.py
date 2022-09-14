# %%
import os
import time
from tqdm import tqdm
# import tqdm
# import tqdm.asyncio
import numpy as np
import soundfile as sf
import pickle

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

from pesq import pesq
with open('/workspace/SE_2022/train_map.pkl', 'rb') as f:
    train_map = pickle.load(f)

def get_pesq(noise_file_path, clean_file_path):
    noise, _ = sf.read(noise_file_path)
    clean, rate = sf.read(clean_file_path)
    return pesq(rate, clean, noise, 'wb')

# %%
# %%
root = '/workspace/data/train'
output_root = '/workspace/output_data/metricgan/train'
truth_score = []
name_list = os.listdir(root)
error_file_list = []

start = time.perf_counter()
for noise_file_path in tqdm(list(train_map.keys())):
    clean_file_name = os.path.basename(train_map[noise_file_path])
    clean_file_path = os.path.join(output_root, clean_file_name)
    try:
        score = get_pesq(noise_file_path, clean_file_path)
        truth_score.append(score)
    except:
        error_file_list.append(noise_file_path)
print(time.perf_counter() - start)
print('model_score_ave:', np.mean(truth_score))

with open('model_score.npy', 'wb') as f:
    np.save(f, np.array(truth_score))
with open('model_score_error_name.npy', 'wb') as f:
    np.save(f, np.array(error_file_list))