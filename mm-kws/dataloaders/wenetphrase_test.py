import torch
from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torchaudio
import pickle
import numpy as np
from torchaudio.compliance.kaldi import fbank
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re

class WenetPhrase_Test_Dataset(Dataset):
    def __init__(
        self,
        test_dir="/nvme01/aizq/mmkws/datasets/WenetPhrase_Clips/WenetPhrase2/S",
        csv="/nvme01/aizq/mmkws/datasets/WenetPhrase_Clips/wenetphrase_test.csv",
        test_text_embedding="/nvme01/aizq/mmkws/datasets/WenetPhrase_Clips/zh_test_text_embeddings.pickle",
        types='easy'
    ):
        self.data = pd.read_csv(csv)
        if types == 'easy':
            self.data = self.data.loc[self.data['type'].isin(['easy', 'pos'])]
        elif types == 'hard':
            self.data = self.data.loc[self.data['type'].isin(['hard', 'pos'])]
        self.data = self.data.values.tolist()
        print(len(self.data))
        with open(test_text_embedding, 'rb') as pickle_file: self.text_embedder = pickle.load(pickle_file)
        self.test_dir = test_dir

    def __getitem__(
        self,
        index
    ):
        # Query_wav_fbank, phoneme, g2p_embed. lm_embed, audiolm_embed, label
        Query_text, Query_wav, Support_text, Support_wav, label, _ = self.data[index]
        # print(Query_text, Query_wav, Support_text, Support_wav, label)
        Query_wav, _ = torchaudio.load(os.path.join(self.test_dir, Query_wav)) # waveform -> fbank
        phoneme = self.text_embedder[Support_text]['phoneme']
        g2p_embed = self.text_embedder[Support_text]['g2p_embed']
        lm_embed = self.text_embedder[Support_text]['lm_embed']
        audiolm_embed = np.load(os.path.join(self.test_dir, Support_wav)[:-4] + '_18.npy')
        fbank_feature = fbank(
            Query_wav,
            num_mel_bins=80
        )
        g2p_embed = torch.from_numpy(g2p_embed)
        g2p_embed = g2p_embed.type_as(fbank_feature)
        return fbank_feature, g2p_embed, torch.from_numpy(lm_embed).squeeze(0), torch.from_numpy(audiolm_embed).squeeze(0), torch.tensor(label)
    

    def __len__(
        self
    ):
        return len(self.data)


def collate_fn(batch):
    # 将 batch 中的每个样本按照其数据类型分组
    fbank_feature, g2p_embed, lm_embed, audiolm_embed, label = zip(*batch)
    # 对每个特征进行填充
    padded_fbank_feature = pad_sequence(fbank_feature, batch_first=True)
    lengths = [len(seq) for seq in padded_fbank_feature]
    padded_g2p_embed = pad_sequence(g2p_embed, batch_first=True)
    padded_lm_embed = pad_sequence(lm_embed, batch_first=True).squeeze(0)
    padded_audiolm_embed = pad_sequence(audiolm_embed, batch_first=True).squeeze(0)
    # 对 label 进行转换为 Tensor
    label_tensor = torch.tensor(label)
    return padded_fbank_feature, torch.tensor(lengths), padded_g2p_embed, padded_lm_embed, padded_audiolm_embed, label_tensor





# test_data = WenetPhrase_Test_Dataset()
# print(test_data[0])
# # dataloader = DataLoader(test_data, batch_size=2, collate_fn=collate_fn, num_workers=1, shuffle=False)
# # from tqdm import tqdm
# # for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
# #     padded_fbank_feature, lengths, padded_g2p_embed, padded_lm_embed, padded_audiolm_embed, label_tensor, dist_tensor = data
# #     print(dist_tensor)
# #     break
# #     # pass
# #     pass