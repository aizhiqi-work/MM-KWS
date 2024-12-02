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

class LibriPhrase_Test_Dataset(Dataset):
    def __init__(
        self,
        test_dir="/nvme01/aizq/mmkws/datasets/LibriPhrase_Test",
        csv=[
            "libriphrase_diffspk_all_1word.csv",
            "libriphrase_diffspk_all_2word.csv",
            "libriphrase_diffspk_all_3word.csv",
            "libriphrase_diffspk_all_4word.csv"
        ],
        save_path='/nvme01/aizq/mmkws/datasets/LibriPhrase_Test/test_phrase.csv',
        test_text_embedding="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_MIN_20/test_text_embeddings.pickle",
        preprocess=False,
        types='easy'
    ):
        if preprocess:
            self.data = pd.DataFrame(columns=['Query_text', 'Query_wav', 'Query_dur', 'Support_text', 'Support_wav', 'Support_dur', 'label', 'type'])
            for path in csv:
                n_word = os.path.join(test_dir, path)
                df = pd.read_csv(n_word)
                anc = df[['anchor_text', 'anchor', 'anchor_dur', 'comparison_text', 'comparison', 'comparison_dur', 'target', 'type']]
                com = df[['comparison_text', 'comparison', 'comparison_dur', 'anchor_text', 'anchor', 'anchor_dur', 'target', 'type']]
                self.data = self.data._append(anc.rename(columns={y: x for x, y in zip(self.data.columns, anc.columns)}), ignore_index=True)
                self.data = self.data._append(com.rename(columns={y: x for x, y in zip(self.data.columns, com.columns)}), ignore_index=True)
            self.data.to_csv(save_path, index=False)
        else:
            self.data = pd.read_csv(save_path)
        # print(self.data)/
        # self.data['dist'] = self.data.apply(lambda x: Levenshtein.ratio(re.sub(r"[^a-zA-Z0-9]+", ' ', x['Support_text']), re.sub(r"[^a-zA-Z0-9]+", ' ', x['Query_text'])), axis=1)
        if types == 'easy':
            self.data = self.data.loc[self.data['type'].isin(['diffspk_easyneg', 'diffspk_positive'])]
        elif types == 'hard':
            self.data = self.data.loc[self.data['type'].isin(['diffspk_hardneg', 'diffspk_positive'])]
        

        self.data = self.data.values.tolist()
        # self.data = self.data[:1000]
        with open(test_text_embedding, 'rb') as pickle_file: self.text_embedder = pickle.load(pickle_file)
        self.test_dir = test_dir

    def __getitem__(
        self,
        index
    ):
        # Query_wav_fbank, phoneme, g2p_embed. lm_embed, audiolm_embed, label
        Query_text, Query_wav, _, Support_text, Support_wav, _, label, _ = self.data[index]
        # print(Query_text, Query_wav, Support_text, Support_wav, label)
        Query_wav, _ = torchaudio.load(os.path.join(self.test_dir, Query_wav)) # waveform -> fbank
        phoneme = self.text_embedder[Support_text]['phoneme']
        g2p_embed = self.text_embedder[Support_text]['g2p_embed']
        lm_embed = self.text_embedder[Support_text]['lm_embed']
        audiolm_embed = np.load(os.path.join(self.test_dir, Support_wav)[:-4] + '.npy')
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





# test_data = LibriPhrase_Test_Dataset()
# dataloader = DataLoader(test_data, batch_size=128, collate_fn=collate_fn, num_workers=16, shuffle=False)
# from tqdm import tqdm
# for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
#     padded_fbank_feature, lengths, padded_g2p_embed, padded_lm_embed, padded_audiolm_embed, label_tensor, dist_tensor = data
#     print(dist_tensor)
#     break
#     # pass
#     pass