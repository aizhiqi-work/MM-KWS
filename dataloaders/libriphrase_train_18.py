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


def get_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav')):
                audio_files.append(os.path.join(root, file))
    return audio_files

class LibriPhrase_Train_Dataset(Dataset):
    def __init__(
        self,
        data_env="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_MIN_20/data",
        audiolm_env="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_MIN_20_XLRALL/audiolm", # 更换了路径，采用18层
        data_json="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_MIN_20/output_20.json",
        data_pos="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_MIN_20/pos.pickle",
        data_text="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_MIN_20/train_text_embeddings.pickle",
        Query_phoneme = "/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_ALL/Query_phoneme.json",
        noise_pt='/nvme01/aizq/mmkws/datasets/noise_data.pt',
        vits_pos_dir="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_VITS_Pos",
        vits_hard_neg_dir="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_VITS_Neg_Resample",
        hard_neg_target="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_MIN_20/hard_neg_target.json",
        all_data_env="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_ALL/libriphrase_clean",
        all_data_json="/nvme01/aizq/mmkws/datasets/LibriPhrase_Train_ALL/exists.json",
    ):
        self.data_env = lmdb.open(data_env, readonly=True)
        self.data_txn = self.data_env.begin()
        self.audiolm_env = audiolm_env
        # self.audiolm_env = lmdb.open(audiolm_env, readonly=True)
        # self.audiolm_txn = self.audiolm_env.begin()
        with open(data_json, 'rb') as json_file: self.keys = json.load(json_file)
        with open(data_pos, 'rb') as pickle_file: self.pos_data = pickle.load(pickle_file)
        with open(data_text, 'rb') as pickle_file: self.text_embedder = pickle.load(pickle_file)
        with open(Query_phoneme, 'rb') as json_file: self.query_phoneme = json.load(json_file)
        from g2p.g2p_en import G2p
        self.g2p = G2p()
        self.tokenizer = DistilBertTokenizer.from_pretrained('/nvme01/aizq/mmkws/distilbert-base-multilingual-cased')
        self.noise_data = torch.load(noise_pt)

        self.all_env = lmdb.open(all_data_env, readonly=True)
        self.all_txn = self.all_env.begin()
        with open(hard_neg_target, 'rb') as json_file: self.hard_neg = json.load(json_file)
        with open(all_data_json, 'rb') as json_file: self.all_data = json.load(json_file)
        self.vits_pos_dir = vits_pos_dir
        self.vits_hard_neg_dir = vits_hard_neg_dir
        self.hard_neg_lists = set(os.listdir(vits_hard_neg_dir))


    # 1212 重新修改了下minibatch读取流程 ，感觉之前的搞反了嘿嘿嘿
    def get_mini_batch(self, key, matching_words):
        support_keys = [f'{key}__{i}' for i in range(len(self.keys[0][key]))]
        mini_support_keys = random.choices(support_keys, k=3)
        pos_keys = []
        neg_keys = list(self.keys[0].keys())
        for match in matching_words:
            match = match[0]
            pos_keys.extend([f'{match}__{i}' for i in range(len(self.keys[0][match]))])
            neg_keys.remove(match)
        for word in mini_support_keys: 
            if word in pos_keys:
                pos_keys.remove(word)
        mini_batch_pos = random.choices(pos_keys, k=15)
        random_nei_keys = random.choices(neg_keys, k=5)
        mini_batch_neg = []
        for _key in random_nei_keys:
            mini_batch_neg.extend(random.choices([f'{_key}__{i}' for i in range(len(self.keys[0][_key]))], k=1))

        vits_pos = get_audio_files(os.path.join(self.vits_pos_dir, key))
        # print(os.path.join(self.vits_pos_dir, key))
        # print(vits_pos)
        mini_batch_vits_pos = random.choices(vits_pos, k=5)

        hard_neg_keys = list(set(self.hard_neg[key]))
        mini_hard_neg_keys = random.choices(hard_neg_keys, k=25)
        mini_batch_vits_hard_neg_keys = [p for p in mini_hard_neg_keys if p in self.hard_neg_lists]
        mini_real_hard_neg_keys = list(set(mini_hard_neg_keys) - set(mini_batch_vits_hard_neg_keys))
        mini_batch_vits_hard_neg_keys = [word for word in mini_batch_vits_hard_neg_keys if all(len(subword) <= 15 for subword in word.split()) and len(word.split()) < 6]
        mini_batch_vits_hard_neg_keys = random.choices(mini_batch_vits_hard_neg_keys, k=min(15, len(mini_batch_vits_hard_neg_keys)))
        mini_real_hard_neg_keys = random.choices(mini_real_hard_neg_keys, k=15-len(mini_batch_vits_hard_neg_keys))
        mini_batch_vits_hard_neg = []
        for _key in mini_batch_vits_hard_neg_keys:
            mini_batch_vits_hard_neg.extend(random.choices(get_audio_files(os.path.join(self.vits_hard_neg_dir, _key)), k=1))
        mini_real_hard_neg = []
        for _key in mini_real_hard_neg_keys:
            mini_real_hard_neg.extend(random.choices([f'{_key}__{i}' for i in range(len(self.all_data[0][_key]))], k=1))

        Query_text, Query_wav, Support_text, Support_wav, label = [], [], [], [], []
        
        _support_keys = [key for _ in mini_support_keys]
        _support_wavs = [self.key2audiolm(support) for support in mini_support_keys]
        Support_text.extend(_support_keys * (len(mini_batch_pos) + len(mini_batch_vits_pos) + len(mini_batch_neg) + len(mini_batch_vits_hard_neg) + len(mini_real_hard_neg)))
        Support_wav.extend(_support_wavs * (len(mini_batch_pos) + len(mini_batch_vits_pos) + len(mini_batch_neg) + len(mini_batch_vits_hard_neg) + len(mini_real_hard_neg)))

        _mini_batch_pos_keys = [pos.split('__')[0] for pos in mini_batch_pos]
        _mini_batch_pos_wavs = [fbank(self.key2wav(pos), num_mel_bins=80) for pos in mini_batch_pos]
        _mini_batch_pos_label = [1 for _ in mini_batch_pos]
        Query_text.extend(_mini_batch_pos_keys * len(mini_support_keys))
        Query_wav.extend(_mini_batch_pos_wavs * len(mini_support_keys))
        label.extend(_mini_batch_pos_label * len(mini_support_keys))

        _mini_batch_vits_pos_keys = [key for _ in mini_batch_vits_pos]
        _mini_batch_vits_pos_wavs = [fbank(self.path2wav(pos), num_mel_bins=80) for pos in mini_batch_vits_pos]
        _mini_batch_vits_pos_label = [1 for _ in mini_batch_vits_pos]
        Query_text.extend(_mini_batch_vits_pos_keys * len(mini_support_keys))
        Query_wav.extend(_mini_batch_vits_pos_wavs * len(mini_support_keys))
        label.extend(_mini_batch_vits_pos_label * len(mini_support_keys))

        _mini_batch_neg_keys = [neg.split('__')[0] for neg in mini_batch_neg]
        _mini_batch_neg_wavs = [fbank(self.key2wav(neg), num_mel_bins=80) for neg in mini_batch_neg]
        _mini_batch_neg_label = [0 for _ in mini_batch_neg]
        Query_text.extend(_mini_batch_neg_keys * len(mini_support_keys))
        Query_wav.extend(_mini_batch_neg_wavs * len(mini_support_keys))
        label.extend(_mini_batch_neg_label * len(mini_support_keys))


        _mini_batch_vits_hard_neg_keys = [neg for neg in mini_batch_vits_hard_neg_keys]
        _mini_batch_vits_hard_neg_wavs = [fbank(self.path2wav(neg), num_mel_bins=80) for neg in mini_batch_vits_hard_neg]
        _mini_batch_vits_hard_neg_label = [0 for _ in mini_batch_vits_hard_neg]
        Query_text.extend(_mini_batch_vits_hard_neg_keys * len(mini_support_keys))
        Query_wav.extend(_mini_batch_vits_hard_neg_wavs * len(mini_support_keys))
        label.extend(_mini_batch_vits_hard_neg_label * len(mini_support_keys))

        _mini_real_hard_neg_keys = [neg.split('__')[0] for neg in mini_real_hard_neg]
        _mini_real_hard_neg_wavs = [fbank(self.key2wav_hard(neg), num_mel_bins=80) for neg in mini_real_hard_neg]
        _mini_real_hard_neg_label = [0 for _ in mini_real_hard_neg]
        Query_text.extend(_mini_real_hard_neg_keys * len(mini_support_keys))
        Query_wav.extend(_mini_real_hard_neg_wavs * len(mini_support_keys))
        label.extend(_mini_real_hard_neg_label * len(mini_support_keys))
        return Query_text, Query_wav, Support_text, Support_wav, label


    def key2wav(self, key):
        audio_data = self.data_txn.get(key.encode())
        audio_bytesio = BytesIO(audio_data)
        audio_numpy = np.frombuffer(audio_bytesio.read(), dtype=np.int16) / 32768
        waveform = torch.from_numpy(audio_numpy.astype(np.float32)).unsqueeze(0)
        if random.random() <= 0.5:
            waveform = self._mixing_snr(waveform)
        return waveform

    def key2wav_hard(self, key):
        audio_data = self.all_txn.get(key.encode())
        audio_bytesio = BytesIO(audio_data)
        audio_numpy = np.frombuffer(audio_bytesio.read(), dtype=np.int16) / 32768
        waveform = torch.from_numpy(audio_numpy.astype(np.float32)).unsqueeze(0)
        if random.random() <= 0.5:
            waveform = self._mixing_snr(waveform)
        return waveform

    def key2audiolm(self, key):
        with open(os.path.join(self.audiolm_env, key.split('__')[0]) + '/' + key + '.bin', "rb") as file:
            audio_data = file.read()
        audiolm = torch.from_numpy(lilcom.decompress(audio_data)).squeeze(0)
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
        key = self.pos_data[index]['key'] # {'key': 'maxineff', 'count': 1, 'matching_words': [('maxineff',)]}
        Query_text, Query_wav, Support_text, Support_wav, label = self.get_mini_batch(key, self.pos_data[index]['matching_words'])
        g2p_embed = []
        lm_embed = []
    
        for text in Support_text:
            g2p_embed.append(self.text_embedder[text]['g2p_embed'])
            lm_embed.append(self.text_embedder[text]['lm_embed'])


        re_text = re.sub(r"[^a-zA-Z0-9]+", ' ', key)
        support_phoneme = self.g2p(re_text)

        Query_phoneme = []
        for text in Query_text:
            Query_phoneme.append(self.query_phoneme[text])
        
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
        
        redo_text = re_text.split(' ')
        support_text_lng = [len(l) - 2 for l in self.tokenizer(redo_text)['input_ids']]
        Text_label = []
        for query_text in Query_text:
            _label = [-1]
            sub_words = query_text.split(' ')
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

    
    def close(self):
        self.data_env.close()
        self.audiolm_env.close()
        self.all_env.close()


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


# LibriPhrase_Train_Dataset()[0]