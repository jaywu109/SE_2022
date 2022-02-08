
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0 , 2'
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
import torch
torch.cuda.set_device(args.local_rank)
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

data_dir = '/workspace/data/train'
train_path_array = get_noise_clean_path(data_dir, train_noise, noise_clean_map)
val_path_array = get_noise_clean_path(data_dir, val_noise, noise_clean_map)
train_dataset = SEDataset(train_path_array, args.batch_size)
val_dataset = SEDataset(val_path_array, args.batch_size)
train_dataloader = SEDataLoader(data_set=train_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                pin_memory=True)
val_dataloader = SEDataLoader(data_set=val_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                pin_memory=True)
data = {'tr_loader': train_dataloader, 'cv_loader': val_dataloader}

optimizer = torch.optim.Adam(model.parameters(),
                                args.lr,
                                weight_decay=args.l2)
solver = Solver(data, model, optimizer, args)

solver.train()


