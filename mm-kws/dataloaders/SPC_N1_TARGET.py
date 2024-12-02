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
import Levenshtein
import re
import random

import os

def get_files(path, endswith):
    _files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(endswith):
                _files.append(os.path.join(root, file))
    return _files
from tqdm import tqdm

class SPC_N1_Dataset(Dataset):
    def __init__(
            self,
            test_dir="/nvme01/aizq/mmkws/mmkws_submits/spc/SPC1/data",
            test_list="/nvme01/aizq/mmkws/mmkws_submits/spc/SPC1/test/text",
            save_path="/nvme01/aizq/mmkws/mmkws_submits/spc/SPC1/SPC_N1_TARGET.csv",
            test_text_embedding="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_MIN_20/spc_text_embeddings.pickle",
            preprocess=True,
        ):
        super().__init__()
        if preprocess:
            target_dict = {}
            idx = 0
            self.data = pd.DataFrame(columns=['id', 'Query_text', 'Query_wav', 'Support_text', 'Support_wav', 'Query_label', 'Support_label', 'label'])
            wav_id, _ = zip(*(line.strip().split() for line in open(test_list)))
            classes = os.listdir(test_dir)
            random.shuffle(classes)
            Targets = classes[:10]
            # Targets = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
            supports_wavs = {}
            for comparison_text in Targets:
                supports_wav = get_files(os.path.join(test_dir, comparison_text), '_18.npy')
                random.shuffle(supports_wav)
                supports_wavs[comparison_text] =  supports_wav
            for wav_idx in range(len(wav_id)):
                wav = os.path.join(test_dir, *wav_id[wav_idx].split('_', 1)) + '.wav'
                query_text = wav_id[wav_idx].split('_')[0]
                if query_text in Targets:
                    for comparison_text in Targets:
                        _label = 1 if comparison_text == query_text else 0
                        target_dict[idx] = {
                            'id': wav_id[wav_idx],
                            'Query_text': query_text,
                            'Query_wav': wav,
                            'Support_text': comparison_text,
                            'Support_wav': supports_wavs[comparison_text][0],
                            'Query_label': Targets.index(query_text),
                            'Support_label':  Targets.index(comparison_text),
                            'label': _label
                        }
                        idx += 1
            self.data = self.data._append(pd.DataFrame.from_dict(target_dict, 'index'), ignore_index=True)
            self.data.to_csv(save_path, index=False)
        else:
            self.data = pd.read_csv(save_path)
        
        self.data = self.data.values.tolist()
        with open(test_text_embedding, 'rb') as pickle_file: self.text_embedder = pickle.load(pickle_file)


    def __getitem__(
        self,
        index
    ):
        ids, Query_text, Query_wav, Support_text, Support_wav, Query_label, Support_label, label = self.data[index]
        Query_wav, _ = torchaudio.load(Query_wav) # waveform -> fbank
        g2p_embed = self.text_embedder[Support_text]['g2p_embed']
        lm_embed = self.text_embedder[Support_text]['lm_embed']
        audiolm_embed = np.load(Support_wav)
        fbank_feature = fbank(
            Query_wav,
            num_mel_bins=80
        )
        g2p_embed = torch.from_numpy(g2p_embed)
        g2p_embed = g2p_embed.type_as(fbank_feature)
        return ids, fbank_feature, g2p_embed, torch.from_numpy(lm_embed).squeeze(0), torch.from_numpy(audiolm_embed).squeeze(0), Query_label, Support_label, torch.tensor(label)

    def __len__(
        self
    ):
        return len(self.data)
    




def collate_fn(batch):
    ids, fbank_feature, g2p_embed, lm_embed, audiolm_embed, Query_label, Support_label, label = zip(*batch)
    padded_fbank_feature = pad_sequence(fbank_feature, batch_first=True)
    lengths = [len(seq) for seq in padded_fbank_feature]
    padded_g2p_embed = pad_sequence(g2p_embed, batch_first=True)
    padded_lm_embed = pad_sequence(lm_embed, batch_first=True).squeeze(0)
    padded_audiolm_embed = pad_sequence(audiolm_embed, batch_first=True).squeeze(0)
    label_tensor = torch.tensor(label)
    return ids, padded_fbank_feature, torch.tensor(lengths), padded_g2p_embed, padded_lm_embed, padded_audiolm_embed, Query_label, Support_label, label_tensor



spc = SPC_N1_Dataset()
# spc[0]