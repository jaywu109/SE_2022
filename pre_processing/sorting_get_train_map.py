# %%
import soundfile as sf
import torchaudio
from pesq import pesq
import random
import os
import numpy as np
import pickle
from collections import Counter
from pprint import pprint

# %% [markdown]
# ### noise type statistics

# %%
def get_noise_stat(noise_type_list):
    noise_type = np.array(list(Counter(noise_type_list).keys()))
    count =  np.array(list(Counter(noise_type_list).values()))
    type_count = np.rec.fromarrays((noise_type, count), names=('noise_type', 'count'))
    pprint(type_count[np.argsort(-1*count)].tolist())

# %%
root = '/workspace/SE/data/train'
train_noise_type_list = []

for flac_name in os.listdir(root):
    if flac_name.endswith('.flac'):
        if flac_name.split('_')[0] == 'mixed':
            train_noise_type_list.append(flac_name.split('.')[0][12:])
print('train_noise_type_stat:')
get_noise_stat(train_noise_type_list)

# %%
root = '/workspace/SE/data/test'
train_noise_type_list = []

for flac_name in os.listdir(root):
    if flac_name.endswith('.flac'):
        if flac_name.split('_')[0] == 'mixed':
            train_noise_type_list.append(flac_name.split('.')[0][12:])
print('test_noise_type_stat:')
get_noise_stat(train_noise_type_list)

# %% [markdown]
# ### Get path for different noise type 

# %%
noise_type = np.array(list(Counter(train_noise_type_list).keys()))
init_list = [[] for _ in noise_type]
noise_type_dict = dict(zip(noise_type, init_list))

root = '/workspace/SE/data/train'

for flac_name in os.listdir(root):
    if flac_name.endswith('.flac'):
        if flac_name.split('_')[0] == 'mixed':
            noise_type_dict[flac_name.split('.')[0][12:]].append(flac_name)
with open('/workspace/SE/SE_2022/train_all_noise_by_type.pkl', 'wb') as f:
    pickle.dump(noise_type_dict, f)

train_path_dict = {}
val_path_dict = {}    

for key in list(noise_type_dict.keys()):
    random.seed(0)
    path_array = np.array(noise_type_dict[key])
    num_of_sample = path_array.shape[0]
    val_index = random.sample(range(num_of_sample), int(num_of_sample*0.1))
    val_path_dict[key] = path_array[val_index]
    train_path_dict[key] = np.delete(path_array, val_index)

with open('/workspace/SE/SE_2022/train_noise_by_type.pkl', 'wb') as f:
    pickle.dump(train_path_dict, f)

with open('/workspace/SE/SE_2022/val_noise_by_type.pkl', 'wb') as f:
    pickle.dump(val_path_dict, f)    

# %% [markdown]
# ### Get the map of noise file to clean file mapping for train data

# %%
root = '/workspace/SE/data/train'
train_map = {}

for flac_name in os.listdir(root):
    if flac_name.endswith('.flac'):
        if flac_name.split('_')[0] == 'mixed':
            number = flac_name.split('_')[1]
            clean_file_name = 'vocal_'+ number + '.flac'
            noise_file_path = os.path.join(root, flac_name)
            clean_file_path = os.path.join(root, clean_file_name)
            train_map[noise_file_path] = clean_file_path

# %%
len(train_map)

# %%
with open('/workspace/SE/SE_2022/train_map.pkl', 'wb') as f:
    pickle.dump(train_map, f)

# %%
with open('/workspace/SE/SE_2022/train_map.pkl', 'rb') as f:
    train_map = pickle.load(f)

# %%
train_map


