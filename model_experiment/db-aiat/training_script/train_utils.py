import argparse
import os 
import numpy as np 
import soundfile as sf
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser("gated complex convolutional recurrent neural network")

is_conti = True
batch_size = 3
epochs = 40
lr = 3e-4
half_lr = 1
early_stop = 40
shuffle = 1
num_workers = 3
print_freq = 10
l2 = 1e-7
model_best_path = '/workspace/model_log/db-aiat/by_type/blower_new/BEST_MODEL/best.pth.tar'
check_point_path = '/workspace/model_log/db-aiat/by_type/blower_new/CP_dir'
loss_dir = '/workspace/model_log/db-aiat/by_type/blower_new/LOSS/loss.mat'

conti_path = '/workspace/SE_2022/model_experiment/db-aiat/DB-AIAT/BEST_MODEL/vb_aia_merge_new.pth.tar'
json_dir = '/home/yuguochen/vbdata/Json'
file_path = '/home/yuguochen/vbdataset'


os.makedirs(os.path.dirname(model_best_path), exist_ok=True)
os.makedirs(os.path.dirname(loss_dir), exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)

parser.add_argument('--json_dir', type=str, default=json_dir,
                    help='The directory of the dataset feat,json format')
parser.add_argument('--loss_dir', type=str, default=loss_dir,
                    help='The directory to save tr loss and cv loss')
parser.add_argument('--batch_size', type=int, default=batch_size,
                    help='The number of the batch size')
parser.add_argument('--cv_batch_size', type=int, default=batch_size,
                    help='The number of the batch size')
parser.add_argument('--epochs', type=int, default=epochs,
                    help='The number of the training epoch')
parser.add_argument('--lr', type=float, default=lr,
                    help='Learning rate of the network')
parser.add_argument('--early_stop', dest='early_stop', default=early_stop, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--half_lr', type=int, default=half_lr,
                    help='Whether to decay learning rate to half scale')
parser.add_argument('--shuffle', type=int, default=shuffle,
                    help='Whether to shuffle within each batch')
parser.add_argument('--num_workers', type=int, default=num_workers,
                    help='Number of workers to generate batch')
parser.add_argument('--l2', type=float, default=l2,
                    help='weight decay (L2 penalty)')
parser.add_argument('--best_path', default=model_best_path,
                    help='Location to save best cv model')
parser.add_argument('--cp_path', type=str, default=check_point_path)
parser.add_argument('--print_freq', type=int, default=print_freq,
                    help='The frequency of printing loss infomation')
parser.add_argument('--is_conti', type=bool, default=is_conti)
parser.add_argument('--conti_path', type=str, default=conti_path)
parser_all = parser

def get_noise_clean_path(data_dir, noise_map, noise_clean_map):
    noise_path_list = []
    clean_path_list = []
    for path_list in [*noise_map.values()]:
        for name in path_list:
            noise_path = os.path.join(data_dir, name)
            noise_path_list.append(noise_path)
            clean_path = noise_clean_map[noise_path]
            clean_path_list.append(clean_path)

    random.seed(0)
    shuffle_index = list(range(len(noise_path_list)))
    random.shuffle(shuffle_index)
    return np.array([np.array(noise_path_list)[shuffle_index], np.array(clean_path_list)[shuffle_index]]).T    

class To_Tensor(object):
    def __call__(self, x, type='float'):
        if type == 'float':
            return torch.FloatTensor(x)
        elif type == 'int':
            return  torch.IntTensor(x)

class BatchInfo(object):
    def __init__(self, feats, labels, frame_mask_list):
        self.feats = feats
        self.labels = labels
        self.frame_mask_list = frame_mask_list

class SEDataset(Dataset): # can pre-process the data here. and don't have to set up minibatch
    def __init__(self, path_array, batch_size):
        self.batch_size = batch_size
        minibatch = []
        start = 0
        while True:
            end = min(len(path_array), start+ batch_size)
            minibatch.append(path_array[start:end])
            start = end
            if end == len(path_array):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]     

class SEDataLoader(object):
    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(dataset=data_set,
                                      shuffle=1,
                                      collate_fn=self.collate_fn,
                                      **kw)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader

def generate_feats_labels(batch):
    win_size = 320
    fft_num = 320
    win_shift = 160
    chunk_length = 3*16000    
    batch = batch[0]
    feat_list, label_list, frame_mask_list = [], [], []
    to_tensor = To_Tensor()
    for id in range(len(batch)):
        noise_path, clean_path = batch[id]
        feat_wav, _= sf.read(noise_path)
        label_wav, _ = sf.read(clean_path)
        c = np.sqrt(len(feat_wav) / np.sum(feat_wav ** 2.0))
        feat_wav, label_wav = to_tensor(feat_wav * c), to_tensor(label_wav * c)
        if len(feat_wav) > chunk_length:
            wav_start = random.randint(0, len(feat_wav)- chunk_length)
            feat_wav = feat_wav[wav_start:wav_start + chunk_length]
            label_wav = label_wav[wav_start:wav_start + chunk_length]

        frame_num = (len(feat_wav) - win_size + fft_num) // win_shift + 1
        frame_mask_list.append(frame_num)
        feat_list.append(feat_wav)
        label_list.append(label_wav)

    feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    feat_list = torch.stft(feat_list, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                           window=torch.hann_window(fft_num)).permute(0,3,2,1) #B F T C-> B C T F
    label_list = torch.stft(label_list, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                            window=torch.hann_window(fft_num)).permute(0,3,2,1)
    return feat_list, label_list, frame_mask_list   