import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import json
import pickle
import lmdb
import random
import numpy as np
from io import BytesIO
from torchaudio.compliance.kaldi import fbank
import lilcom
import torchaudio
import re
from transformers import DistilBertTokenizer
import warnings
warnings.filterwarnings('ignore')


def get_files(path, endswith=['.wav']):
    filepaths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in endswith):
                filepath = os.path.join(dirpath, filename)
                filepaths.append(filepath)
    return filepaths

class WenetPhrase_Train_Dataset(Dataset):
    def __init__(
        self,
        data_env="/nvme01/aizq/mmkws/datasets/WenetPhrase_Clips/WenetPhrase2/M_S",
        audiolm_env="/nvme01/aizq/mmkws/datasets/WenetPhrase_Clips/WenetPhrase2AudioLM/M_S",
        data_json="/nvme01/aizq/mmkws/datasets/WenetPhrase_Clips/wenetphrase_pos.json",
        data_text="/nvme01/aizq/mmkws/datasets/WenetPhrase_Clips/zh_train_text_embeddings.pickle",
        noise_pt='/nvme01/aizq/mmkws/datasets/noise_data.pt',
        vits_pos_dir="/nvme01/aizq/mmkws/datasets/WenetPhrase_Clips/WenetPhrase2VitsPos_Resample",
        vits_hard_neg_dir="/nvme01/aizq/mmkws/datasets/WenetPhrase_Clips/WenetPhrase2VitsNeg_Resample",
        hard_neg_target="/nvme01/aizq/mmkws/funasr/hard_neg_dataset.json"
    ):
        self.data_env = data_env
        self.audiolm_env = audiolm_env
        with open(data_json, 'rb') as json_file: self.keys = list(json.load(json_file).keys())
        with open(data_json, 'rb') as json_file: self.pos_data = json.load(json_file)
        with open(data_text, 'rb') as pickle_file: self.text_embedder = pickle.load(pickle_file)
        from g2pM import G2pM
        self.g2pm = G2pM()
        self.tokenizer = DistilBertTokenizer.from_pretrained('/nvme01/aizq/mmkws/distilbert-base-multilingual-cased')
        self.noise_data = torch.load(noise_pt)
        self.all_keys = os.listdir("/nvme01/aizq/mmkws/datasets/WenetPhrase_Clips/WenetPhrase2/M_S")
        
        with open(hard_neg_target, 'rb') as json_file: self.hard_neg = json.load(json_file)
        self.vits_pos_dir = vits_pos_dir
        self.vits_hard_neg_dir = vits_hard_neg_dir
        self.all_hard_keys = os.listdir(vits_hard_neg_dir)
        self.hard_neg_lists = set(os.listdir(vits_hard_neg_dir))
        self.data_env_lists = set(os.listdir(data_env))
        

                
    # 1212 重新修改了下minibatch读取流程 ，感觉之前的搞反了嘿嘿嘿
    def get_mini_batch(self, key, matching_words):
        support_wavs = get_files(os.path.join(self.data_env, key), '.wav')
        mini_support_wavs = random.choices(support_wavs, k=3)
        pos_wavs = []
        neg_keys = self.all_keys
        for match in matching_words:
            pos_wavs.extend(get_files(os.path.join(self.data_env, match), '.wav'))
            if match in neg_keys:
                neg_keys.remove(match)      
        for word in mini_support_wavs: 
            if word in pos_wavs:
                pos_wavs.remove(word)
        mini_batch_pos = random.choices(pos_wavs, k=8)

        
        vits_pos = get_files(os.path.join(self.vits_pos_dir, key), '.wav')
        mini_batch_vits_pos = random.choices(vits_pos, k=2)
        hard_neg_keys = list(set(self.hard_neg[key]))
        mini_hard_neg_keys = hard_neg_keys
        mini_real_hard_neg_keys = [p for p in mini_hard_neg_keys if p in self.data_env_lists]
        mini_batch_vits_hard_neg_keys = [p for p in mini_hard_neg_keys if p in self.hard_neg_lists]        
        try:
            mini_real_hard_neg_keys = random.choices(mini_real_hard_neg_keys, k=min(6, len(mini_real_hard_neg_keys)))
            mini_batch_vits_hard_neg_keys = random.choices(mini_batch_vits_hard_neg_keys, k=min(6-len(mini_real_hard_neg_keys), len(mini_batch_vits_hard_neg_keys)))
        except:
            print(key)
            raise 'error'
        random_nei_keys = random.choices(neg_keys, k=10-len(mini_real_hard_neg_keys)-len(mini_batch_vits_hard_neg_keys))
        mini_batch_neg = []
        for _key in random_nei_keys:
            mini_batch_neg.extend(random.choices(get_files(os.path.join(self.data_env, _key), '.wav'), k=1))        
        

        
        mini_batch_vits_hard_neg = []
        for _key in mini_batch_vits_hard_neg_keys:
            mini_batch_vits_hard_neg.extend(random.choices(get_files(os.path.join(self.vits_hard_neg_dir, _key)), k=1))
        mini_real_hard_neg = []
        for _key in mini_real_hard_neg_keys:
            mini_real_hard_neg.extend(random.choices(get_files(os.path.join(self.data_env, _key)), k=1))
            
        Query_text, Query_wav, Support_text, Support_wav, label = [], [], [], [], []
        
        # print(mini_batch_pos)
        # print(mini_batch_neg)
        # print(mini_batch_vits_pos)
        # print(mini_batch_vits_hard_neg)
        # print(mini_real_hard_neg)
        _support_keys = [key for _ in mini_support_wavs]
        _support_wavs = [self.key2audiolm(support) for support in mini_support_wavs]
        Support_text.extend(_support_keys * (len(mini_batch_pos) + len(mini_batch_vits_pos) + len(mini_batch_neg) + len(mini_batch_vits_hard_neg) + len(mini_real_hard_neg)))
        Support_wav.extend(_support_wavs * (len(mini_batch_pos) + len(mini_batch_vits_pos) + len(mini_batch_neg) + len(mini_batch_vits_hard_neg) + len(mini_real_hard_neg)))

        _mini_batch_pos_keys = [pos.split('/')[-2] for pos in mini_batch_pos]
        _mini_batch_pos_wavs = [fbank(self.path2wav(pos), num_mel_bins=80) for pos in mini_batch_pos]
        _mini_batch_pos_label = [1 for _ in mini_batch_pos]
        Query_text.extend(_mini_batch_pos_keys * len(mini_support_wavs))
        Query_wav.extend(_mini_batch_pos_wavs * len(mini_support_wavs))
        label.extend(_mini_batch_pos_label * len(mini_support_wavs))

        _mini_batch_vits_pos_keys = [vits_pos.split('/')[-2] for vits_pos in mini_batch_vits_pos]
        _mini_batch_vits_pos_wavs = [fbank(self.path2wav(pos), num_mel_bins=80) for pos in mini_batch_vits_pos]
        _mini_batch_vits_pos_label = [1 for _ in mini_batch_vits_pos]
        Query_text.extend(_mini_batch_vits_pos_keys * len(mini_support_wavs))
        Query_wav.extend(_mini_batch_vits_pos_wavs * len(mini_support_wavs))
        label.extend(_mini_batch_vits_pos_label * len(mini_support_wavs))
        
        _mini_batch_neg_keys = [neg.split('/')[-2] for neg in mini_batch_neg]
        _mini_batch_neg_wavs = [fbank(self.path2wav(neg), num_mel_bins=80) for neg in mini_batch_neg]
        _mini_batch_neg_label = [0 for _ in mini_batch_neg]
        Query_text.extend(_mini_batch_neg_keys * len(mini_support_wavs))
        Query_wav.extend(_mini_batch_neg_wavs * len(mini_support_wavs))
        label.extend(_mini_batch_neg_label * len(mini_support_wavs))
        
        _mini_batch_vits_hard_neg_keys = [vits_neg.split('/')[-2] for vits_neg in mini_batch_vits_hard_neg]
        _mini_batch_vits_hard_neg_wavs = [fbank(self.path2wav(neg), num_mel_bins=80) for neg in mini_batch_vits_hard_neg]
        _mini_batch_vits_hard_neg_label = [0 for _ in mini_batch_vits_hard_neg]
        Query_text.extend(_mini_batch_vits_hard_neg_keys * len(mini_support_wavs))
        Query_wav.extend(_mini_batch_vits_hard_neg_wavs * len(mini_support_wavs))
        label.extend(_mini_batch_vits_hard_neg_label * len(mini_support_wavs))    
    
        _mini_real_hard_neg_keys = [real_neg.split('/')[-2] for real_neg in mini_real_hard_neg]
        _mini_real_hard_neg_wavs = [fbank(self.path2wav(neg), num_mel_bins=80) for neg in mini_real_hard_neg]
        _mini_real_hard_neg_label = [0 for _ in mini_real_hard_neg]
        Query_text.extend(_mini_real_hard_neg_keys * len(mini_support_wavs))
        Query_wav.extend(_mini_real_hard_neg_wavs * len(mini_support_wavs))
        label.extend(_mini_real_hard_neg_label * len(mini_support_wavs))
        return Query_text, Query_wav, Support_text, Support_wav, label


    def key2audiolm(self, key):
        audio_data = np.load(key.replace('WenetPhrase2', 'WenetPhrase2AudioLM')[:-4] + '.npy')
        audiolm = torch.from_numpy(audio_data).squeeze(0)
        return audiolm

    def path2wav(self, path):
        waveform, _ = torchaudio.load(path)
        if random.random() <= 0.5:
            waveform = self._mixing_snr(waveform)
        return waveform


    def _mixing_snr(self, clean, snr=[0, 15]):
        def _cal_adjusted_rms(clean_rms, snr):
            a = float(snr) / 20
            noise_rms = clean_rms / (10**a)
            return noise_rms

        def _cal_rms(amp):
            return torch.sqrt(torch.mean(amp**2, dim=-1))

        start = np.random.randint(0, self.noise_data.shape[-1] - clean.shape[-1] )
        divided_noise = self.noise_data[:, start: start + clean.shape[-1]]
        clean_rms = _cal_rms(clean)
        noise_rms = _cal_rms(divided_noise)
        adj_noise_rms = _cal_adjusted_rms(clean_rms, np.random.randint(snr[0], snr[1]))

        adj_noise_amp = divided_noise * (adj_noise_rms / (noise_rms + 1e-7))
        noisy = clean + adj_noise_amp

        if torch.max(noisy) > 1:
            noisy = noisy / torch.max(noisy)

        return noisy
    
    def __getitem__(self, index):
        key = self.keys[index]
        matching_words = self.pos_data[key]['pos'] # {'key': 'maxineff', 'count': 1, 'matching_words': [('maxineff',)]}
        Query_text, Query_wav, Support_text, Support_wav, label = self.get_mini_batch(key, matching_words)
        g2p_embed = []
        lm_embed = []
    
        for text in Support_text:
            g2p_embed.append(self.text_embedder[text]['g2p_embed'])
            lm_embed.append(self.text_embedder[text]['lm_embed'])

        support_phoneme = self.g2pm(key, tone=True, char_split=False)
        Query_phoneme = []
        for text in Query_text:
            Query_phoneme.append(self.g2pm(text, tone=True, char_split=False))
        Phoneme_label = []
        for query_phoneme in Query_phoneme:
            _label = []
            for phon in support_phoneme:
                if phon == ' ':
                    _label.append(1)
                elif phon in query_phoneme:
                    _label.append(1)
                else:
                    _label.append(0)
            Phoneme_label.append(_label)
        
        redo_text = list(key)
        support_text_lng = [len(l) - 2 for l in self.tokenizer(redo_text)['input_ids']]
        Text_label = []
        for query_text in Query_text:
            _label = [-1]
            sub_words = list(query_text)
            for i, text in enumerate(redo_text):
                if text in sub_words:
                    _label.extend([1] * support_text_lng[i])
                else:
                    _label.extend([0] * support_text_lng[i])
            _label.append(-1)
            Text_label.append(_label)
    
        return Query_wav, g2p_embed, lm_embed, Support_wav, label, Phoneme_label, Text_label


    def __len__(self):
        return len(self.pos_data)



# a = WenetPhrase_Train_Dataset()
# # a[0]
# # a[1]
# a[2]
# a[3]
# a[4]
from torch.nn.utils.rnn import pad_sequence


def padding_and_mask(data):
    max_length = max(len(sequence) for sequence in data)
    padded_data = []
    mask = []
    for sequence in data:
        padded_sequence = sequence + [-1] * (max_length - len(sequence))
        padded_data.append(padded_sequence)
        sequence_mask = [1 if x != -1 else 0 for x in sequence]
        sequence_mask += [0] * (max_length - len(sequence))
        mask.append(sequence_mask)
    return padded_data, mask

def train_collate_fn(batch):
    # 将 batch 中的每个样本按照其数据类型分组
    fbank_feature, g2p_embed, lm_embed, audiolm_embed, label, Phoneme_label, Text_label = zip(*batch)
    merged_fbank_feature = [item for sublist in fbank_feature for item in sublist]
    merged_g2p_embed = [torch.from_numpy(item) for sublist in g2p_embed for item in sublist]
    merged_lm_embed = [torch.from_numpy(item).squeeze(0) for sublist in lm_embed for item in sublist]
    merged_audiolm_embed = [item for sublist in audiolm_embed for item in sublist]
    merged_label = [item for sublist in label for item in sublist]
    
    # 添加的两个新loss
    merged_Phoneme_label = [item for sublist in Phoneme_label for item in sublist]
    merged_Text_label = [item for sublist in Text_label for item in sublist]
 
    # 对每个特征进行填充
    padded_fbank_feature = pad_sequence(merged_fbank_feature, batch_first=True)
    lengths = [len(seq) for seq in merged_fbank_feature]
    padded_g2p_embed = pad_sequence(merged_g2p_embed, batch_first=True)
    padded_lm_embed = pad_sequence(merged_lm_embed, batch_first=True).squeeze(0)
    padded_audiolm_embed = pad_sequence(merged_audiolm_embed, batch_first=True).squeeze(0)
    label_tensor = torch.tensor(merged_label)

    # 添加的两个新loss
    padded_phoneme_label, mask_phoneme_label = padding_and_mask(merged_Phoneme_label)
    padded_phoneme_label, mask_phoneme_label = torch.tensor(padded_phoneme_label), torch.tensor(mask_phoneme_label)
    padded_text_label, mask_text_label = padding_and_mask(merged_Text_label)
    padded_text_label, mask_text_label = torch.tensor(padded_text_label), torch.tensor(mask_text_label)

    # 获得数据的总长度
    total_length = len(padded_fbank_feature)
    random_indices = torch.randperm(total_length)
    shuffled_padded_fbank_feature = padded_fbank_feature[random_indices]
    shuffled_padded_g2p_embed = padded_g2p_embed[random_indices]
    shuffled_padded_lm_embed = padded_lm_embed[random_indices]
    shuffled_padded_audiolm_embed = padded_audiolm_embed[random_indices]
    shuffled_label_tensor = label_tensor[random_indices]
    shuffled_padded_g2p_embed = shuffled_padded_g2p_embed.type_as(shuffled_padded_fbank_feature)

    # 添加的两个新loss
    shuffled_padded_phoneme_label = padded_phoneme_label[random_indices]
    shuffled_mask_phoneme_label = mask_phoneme_label[random_indices]
    shuffled_padded_text_label = padded_text_label[random_indices]
    shuffled_mask_text_label = mask_text_label[random_indices]

    return shuffled_padded_fbank_feature, torch.tensor(lengths), shuffled_padded_g2p_embed, \
        shuffled_padded_lm_embed, shuffled_padded_audiolm_embed, shuffled_label_tensor, \
        shuffled_padded_phoneme_label, shuffled_mask_phoneme_label, shuffled_padded_text_label, shuffled_mask_text_label