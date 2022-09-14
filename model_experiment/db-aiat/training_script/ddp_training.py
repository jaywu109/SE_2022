
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import sys
import traceback
# os.environ['CUDA_VISIBLE_DEVICES'] = '0 , 2'
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args_cuda = parser.parse_args()
import torch
torch.cuda.set_device(args_cuda.local_rank)
torch.distributed.init_process_group(backend='nccl')

import numpy as np
import json
import pickle 
import h5py
import math
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import soundfile as sf
import os
from pprint import pprint
from ranger2020 import Ranger

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
from aia_trans import dual_aia_trans_merge_crm
from solver_merge import Solver
from train_utils import parser_all, SEDataset, SEDataLoader, get_noise_clean_path
args = parser_all.parse_args(args = [])
model = dual_aia_trans_merge_crm().cuda()

with open('/workspace/SE_2022/train_noise_by_type.pkl', 'rb') as f:
    train_noise = pickle.load(f)

with open('/workspace/SE_2022/val_noise_by_type.pkl', 'rb') as f:
    val_noise = pickle.load(f)

with open('/workspace/SE_2022/train_map.pkl', 'rb') as f:
    noise_clean_map = pickle.load(f)

### CAHNGE TYPE FOR DIFFERENT SUBSET
data_type = 'blower'  
###

noise_path_list = []
clean_path_list = []
for path in (train_noise[data_type]):
    noise_path = '/workspace/SE/data/train/' + path
    noise_path_list.append(noise_path)
    clean_path_list.append(noise_clean_map[noise_path])
train_path_array = np.array([np.array(noise_path_list), np.array(clean_path_list)]).T

noise_path_list = []
clean_path_list = []
for path in (val_noise[data_type]):
    noise_path = '/workspace/SE/data/train/' + path
    noise_path_list.append(noise_path)
    clean_path_list.append(noise_clean_map[noise_path])
val_path_array = np.array([np.array(noise_path_list), np.array(clean_path_list)]).T


# noise_path_list = []
# clean_path_list = []
# for noise_path in (noise_clean_map.keys()):
#     noise_path_list.append(noise_path)
#     clean_path_list.append(noise_clean_map[noise_path])
# train_path_array = np.array([np.array(noise_path_list), np.array(clean_path_list)]).T

# root = '/workspace/data/test'
# test_path_list = []
# for flac_name in os.listdir(root):
#     if flac_name.endswith('.flac'):
#         test_path_list.append(os.path.join(root, flac_name))
# test_path_array = np.array([np.array(test_path_list), np.array(test_path_list)]).T  

train_dataset = SEDataset(train_path_array, 2)
val_dataset = SEDataset(val_path_array, 3)

train_dataloader = SEDataLoader(data_set=train_dataset,
                                batch_size=1,
                                num_workers=15,
                                pin_memory=True)
val_dataloader = SEDataLoader(data_set=val_dataset,
                                batch_size=1,
                                num_workers=15,
                                pin_memory=True)
data = {'tr_loader': train_dataloader, 'cv_loader': val_dataloader}

optimizer = Ranger(model.parameters(),
                   lr=3e-4, weight_decay=1e-7)
# optimizer = torch.optim.Adam(model.parameters(),
#                                 args.lr,
#                                 weight_decay=args.l2)
solver = Solver(data, model, optimizer, args)

try:
    solver.train()
except Exception as e:
    error_class = e.__class__.__name__ #取得錯誤類型
    detail = e.args[0] #取得詳細內容
    cl, exc, tb = sys.exc_info() #取得Call Stack
    lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
    fileName = lastCallStack[0] #取得發生的檔案名稱
    lineNum = lastCallStack[1] #取得發生的行號
    funcName = lastCallStack[2] #取得發生的函數名稱
    errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
    with open("error_message.txt", "w") as text_file:
        text_file.write(errMsg)

