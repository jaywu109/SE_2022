# %%
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import librosa
import os
import numpy as np
import numpy as npb
from tqdm import tqdm
from istft import ISTFT
from aia_trans import aia_complex_trans_mag, aia_complex_trans_ri, dual_aia_trans_merge_crm
import soundfile as sf

# %%
class Enhance:
    def __init__(self, args):
        self.model = dual_aia_trans_merge_crm()
        checkpoint = torch.load(args['Model_path'])
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.cuda()
        self.istft = ISTFT(filter_length=320, hop_length=160, window='hanning')
        self.fs = args['fs']

    def enhance(self, noise_file_path, clean_file_path):
        with torch.no_grad():
            feat_wav, _ = sf.read(noise_file_path)
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
            fake_wav_len = (frame_num - 1) * 160 + 320 - 320
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
                                window=torch.hann_window(320)).permute(0, 3, 2, 1)
            noisy_phase = torch.atan2(feat_x[:, -1, :, :], feat_x[:, 0, :, :])
            feat_x_mag = (torch.norm(feat_x, dim=1)) ** 0.5
            feat_x = torch.stack((feat_x_mag * torch.cos(noisy_phase), feat_x_mag * torch.sin(noisy_phase)), dim=1)
            esti_x = self.model(feat_x.cuda())
            esti_mag, esti_phase = torch.norm(esti_x, dim=1), torch.atan2(esti_x[:, -1, :, :],
                                                                            esti_x[:, 0, :, :])
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            esti_com = esti_com.cpu()
            esti_utt = self.istft(esti_com).squeeze().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c
            sf.write(clean_file_path, esti_utt, args['fs'])

# %%
def enhance_ri(args, noise_file_path, clean_file_path):
    model = aia_complex_trans_ri()
    checkpoint = torch.load(args['Model_path'])['model_state_dict']
    model.load_state_dict(checkpoint)
    print(model)
    model.eval()
    model.cuda()

    with torch.no_grad():
        cnt = 0
        mix_file_path = noise_file_path
        esti_file_path = clean_file_path
        file_list = os.listdir(mix_file_path)
        istft = ISTFT(filter_length=320, hop_length=160, window='hanning')
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
            fake_wav_len = (frame_num - 1) * 160 + 320 - 320
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
                                window=torch.hann_window(320)).permute(0, 3, 2, 1)
            noisy_phase = torch.atan2(feat_x[:, -1, :, :], feat_x[:, 0, :, :])
            feat_x_mag = (torch.norm(feat_x, dim=1)) ** 0.5
            feat_x = torch.stack((feat_x_mag * torch.cos(noisy_phase), feat_x_mag * torch.sin(noisy_phase)), dim=1)
            esti_x = model(feat_x.cuda())
            esti_mag, esti_phase = torch.norm(esti_x, dim=1), torch.atan2(esti_x[:, -1, :, :],
                                                                             esti_x[:, 0, :, :])
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            esti_com = esti_com.cpu()
            esti_utt = istft(esti_com).squeeze().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c
            os.makedirs(os.path.join(esti_file_path), exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args['fs'])
            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1

# %%
args = {}
args['Model_path'] = '/workspace/SE_2022/model_experiment/db-aiat/DB-AIAT/BEST_MODEL/vb_aia_merge_new.pth.tar'
args['fs'] = 16000
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

enhance_model = Enhance(args)
    

# %% [markdown]
# ### Prediction

# %%
# root = '/workspace/data/test'
# output_dir = '/workspace/output_data/db-aiat/test'

# for flac_name in tqdm(os.listdir(root)):
#     if flac_name.endswith('.flac'):
#         number = flac_name.split('_')[1]
#         clean_file_name = 'vocal_'+ number + '.flac'
#         noise_file_path = os.path.join(root, flac_name)
#         clean_file_path = os.path.join(output_dir, clean_file_name)
#         try:
#             enhance_model.enhance(noise_file_path, clean_file_path)
#         except:
#             print('Error:', flac_name)

# %%
root = '/workspace/data/train'
output_dir = '/workspace/output_data/db-aiat/train'

for flac_name in tqdm(os.listdir(root)):
    if flac_name.endswith('.flac'):
        if flac_name.split('_')[0] == 'mixed':
            number = flac_name.split('_')[1]
            clean_file_name = 'vocal_'+ number + '.flac'
            noise_file_path = os.path.join(root, flac_name)
            clean_file_path = os.path.join(output_dir, clean_file_name)
            try:
                enhance_model.enhance(noise_file_path, clean_file_path)
            except:
                print('Error:', flac_name)


