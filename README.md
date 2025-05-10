# MM-KWS: Multi-modal Prompts for Multilingual User-defined Keyword Spotting

The official implementations of "Multi-modal Prompts for Multilingual User-defined Keyword Spotting" (accepted by Interspeech 2024).

[2024.6][Paper](https://www.isca-archive.org/interspeech_2024/ai24_interspeech.html)

[2025.5]Recently, we have optimized Chinese custom wakeup words, performed framework optimization and larger scale pre-training, achieved significant improvement on no-fine-tuning custom words, and will open source part of the model and the optimized inference model at the end of May -> [OpenKWS](https://github.com/aizhiqi-work/OpenKWS) 


## Introduction

MM-KWS, a novel approach to user-defined keyword spotting leveraging multi-modal enrollments of text and speech templates.

![alt text](<asserts/overview.png>)

## Data-pipeline for WenetPhrase

Please note that the wenetphrase dataset presented in MM-KWS is sliced and diced from the [WenetSpeech](https://arxiv.org/pdf/2110.03370) and is copyrighted by the original data authors.

1. Read raw wenetspeech to wenetclip
    ```
    python data-processing/wenetspeech/read.py
    ```
    Then you can get:
    ```
    wenetspeech_clips
        - M_S   # for train
            - podcast
                - B00000
                    - X0000000000_100638174_S00002.txt
                    - X0000000000_100638174_S00002.wav
                    ...
                - ...
            - youtube
        - S     # for test
            - podcast
                - B00000
                    - X0000000000_100638174_S00037.txt
                    - X0000000000_100638174_S00037.wav
                    ...
                - ...
    ```
2. Norm text, we use [Chinese Norm](https://github.com/Joee1995/chn_text_norm.git)
    ```
    python data-processing/wenetspeech/norm_txt.py
    ```
    then get -xxxx_norm.txt

3. CLIP wenetspeech ~~~ 😄
    ```
    python data-processing/wenetspeech/wenetclip.py
    ```
    In [wenetclip.py](), we use [TORCHAUDIO_MFA]()  and [g2pm]() for transcript
    ```
    from torchaudio.pipelines import MMS_FA as bundle
    ```
    Then you can get:
    ```
    Wenetphrase
        - M_S   # for train
            - 121.4 MiB [##########] /现在
            - 102.2 MiB [########  ] /知道
            - 93.6 MiB [#######   ] /时候
            - 85.6 MiB [#######   ] /孩子
            - 81.6 MiB [######    ] /今天
            - 78.8 MiB [######    ] /事情
            - 74.0 MiB [######    ] /非常
            - 72.0 MiB [#####     ] /为什么
            ...
        - S     # for test
            ...
    ```
    Total disk usage:  40.6 GiB  Apparent size:  35.3 GiB  Items: 3039907
    ```
    36.4 GiB [##########] /M_S
    4.2 GiB [#         ] /S 
    ```
4. MM-KWS [WenetPhrase-test.csv](): https://pan.baidu.com/s/1rJgSwi6fZjHto_wxUHft2w?pwd=auyt 提取码: auyt
5. [WenetPhrase data](): https://pan.baidu.com/s/1aiykUi9aZGHgODIBkXv64g?pwd=r3t6 提取码: r3t6 

### Train Log:
LibriPhrase:
![alt text](<asserts/libriphrase.png>)
WenetPhrase:
![alt text](<asserts/wenetphrase.png>)

### For audiolm and g2p & g2pm:
1. audiolm: we use XLR53 (as a better multilingual capability), we extract to lmdb for offline-extract:
   ```
    core:
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        audiolm = bundle.get_model()
            with torch.inference_mode():
                features, _ = audiolm.extract_features(waveform)
                out_feature = features[17].cpu().detach().numpy() # 18 layer， better than the last.
   ```
2. g2p, we follow [PhonMatchNet](https://arxiv.org/abs/2308.16511):
   ```
   core
    def embedding(self, text):
        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        words = word_tokenize(text)

        # embedding func.
        def _get(self, word):
            # encoder
            enc = self.encode(word)
            enc = self.gru(enc, len(word) + 1, self.enc_w_ih, self.enc_w_hh,
                        self.enc_b_ih, self.enc_b_hh, h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32))
            last_hidden = enc[:, -1, :]

            # decoder
            dec = np.take(self.dec_emb, [2], axis=0)  # 2: <s>
            h = last_hidden

            preds = []
            emb = np.empty((0, self.dec_emb[0,:].shape[-1]))
            for i in range(20):
                h = self.grucell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)  # (b, h)
                logits = np.matmul(h, self.fc_w.T) + self.fc_b
                pred = logits.argmax()
                if pred == 3: break  # 3: </s>
                dec = np.take(self.dec_emb, [pred], axis=0)
                emb = np.append(emb, h, axis=0)

            return emb
        
        # steps
        embed = np.empty((0, self.dec_emb[0,:].shape[-1]))
        for word in words:
            if re.search("[a-z]", word) is None:
                continue
            embed = np.append(embed, _get(self, word), axis=0)
            embed = np.append(embed, np.take(self.dec_emb, [0], axis=0), axis=0)

        return embed[:-1,:]
   ```
3. g2pm, we add embedding code:
   ```
   core
       def embedding(self, sent, char_split=False, tone=True):
        def _get(inputs):
            lengths = np.sum(np.sign(inputs), axis=1)
            max_length = max(lengths)

            rev_seq = self.reverse_sequence(inputs, lengths)
            fw_emb = self.get_embedding(inputs)  # [b,t,d]
            bw_emb = self.get_embedding(rev_seq)

            fw_states, bw_states = None, None
            fw_hs = []
            bw_hs = []
            for i in range(max_length):
                fw_input = fw_emb[:, i, :]
                bw_input = bw_emb[:, i, :]
                fw_states = self.fw_lstm_cell(fw_input, fw_states)
                bw_states = self.bw_lstm_cell(bw_input, bw_states)

                fw_hs.append(fw_states[0])
                bw_hs.append(bw_states[0])
            fw_hiddens = np.stack(fw_hs, axis=1)
            bw_hiddens = np.stack(bw_hs, axis=1)
            bw_hiddens = self.reverse_sequence(bw_hiddens, lengths)
            outputs = np.concatenate([fw_hiddens, bw_hiddens], axis=2)  # [b,t,d]
            return outputs
        input_ids = []
        poly_indices = []
        pros_lst = []
        for idx, char in enumerate(sent):
            if char in self.char2idx:
                char_id = self.char2idx[char]
            else:
                char_id = self.char2idx[UNK_TOKEN]
            input_ids.append(char_id)

            if char in self.cedict:
                prons = self.cedict[char]

                # polyphonic character
                if len(prons) > 1:
                    poly_indices.append(idx)
                    pros_lst.append(SPLIT_TOKEN)
                else:
                    pron = prons[0]
                    # remove the digit which denotes a tone
                    if not tone:
                        pron = pron[:-1]
                    pros_lst.append(pron)
            else:
                pros_lst.append(char)
            
        # insert and append BOS, EOS ID
        BOS_ID = self.char2idx[BOS_TOKEN]
        EOS_ID = self.char2idx[EOS_TOKEN]
        input_ids.insert(0, BOS_ID)
        input_ids.append(EOS_ID)
        input_ids = np.array(input_ids, dtype=np.int32)
        input_ids = np.expand_dims(input_ids, axis=0)     
        embed = np.array(_get(input_ids))[0][1:-1, :]
        return embed
    ```
4. for word embedding, we use [distilbert-base-multilingual-cased] from huggingface.

## Data aug
In Libriphrase, we choise 27k classes (samples >= 20), as anchors [PS: libriphrase train-data from librispeech clean-360/100 and libriphrase test-data from librispeech others-500]().
so we generated the anchor-words as this data-aug pipeline:
![截屏2024-12-03 08 43 34](https://github.com/user-attachments/assets/399e225d-6e27-4730-bfad-baa2bb8927df)
you can get libriphrase_hardneg.json.zip and wenetphrase_hardneg.json.zip in the repo, that's the hard words for the anchor words (27k classes).
PS: In MM-KWS version, we just use random neg. And in MM-KWS* version, we use the hard neg for data aug.

In speech synthesis, we use Vits-based TTS [TSCT-TTS](https://great-research.github.io/tsct-tts-demo/), [ps. our work on TTS, last year] but for now I'd recommend the gpt-sovits or cosyvoice, the newer tts are too powerful 👍👍


## Citation
If you want to cite this paper:
```
@inproceedings{ai24_interspeech,
  title     = {MM-KWS: Multi-modal Prompts for Multilingual User-defined Keyword Spotting},
  author    = {Zhiqi Ai and Zhiyong Chen and Shugong Xu},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {2415--2419},
  doi       = {10.21437/Interspeech.2024-10},
  issn      = {2958-1796},
}
```
