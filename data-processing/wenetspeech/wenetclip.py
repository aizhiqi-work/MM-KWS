import os
import multiprocessing
from tqdm import tqdm


def find_files_with_suffix(root_folder, endwiths):
    matching_files = []  # 以后缀名为键，文件路径列表为值初始化字典
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            for suffix in endwiths:
                if file.endswith(suffix):
                    matching_files.append(os.path.join(root, file))
    return matching_files


def read_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read().rstrip('\n')
    return content


# 创建一个包含汉字数量大于等于2的集合
def select_hanzi_greater_than_or_equal_to_2(input_set):
    result_set = set()
    for item in input_set:
        # 判断是否只包含汉字且汉字数量大于等于2
        if all('\u4e00' <= char <= '\u9fff' for char in item) and len(item) >= 2 and len(item) <= 6:
            result_set.add(item)
    return result_set


import re

def find_all_occurrences(text, pattern):
    occurrences = [(match.start(), match.end()) for match in re.finditer(pattern, text)]
    return occurrences


import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle
from typing import List
from g2pM import G2pM # g2pM # g2pW 论文结果更好，但是是台湾人高的，普通话还需要额外做好多操作emm
import jieba
import numpy as np

def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)



class TORCHAUDIO_MFA:
    def __init__(self, device_id='0', save_dirs="/server24/aizq/mm_kws/datasets/WenetPhrase/M_S"):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.mfa = bundle.get_model()
        self.mfa.to(self.device)
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.g2pm = G2pM()
        self.save_dirs = save_dirs
    

    def compute_alignments(self, waveform: torch.Tensor, transcript: List[str]):
        with torch.inference_mode():
            emission, _ = self.mfa(waveform.to(self.device))
            token_spans = self.aligner(emission[0], self.tokenizer(transcript))
        return emission, token_spans


    def make_mfa(self, wav_file: str, sentence: str):
        try:
            seg_list = select_hanzi_greater_than_or_equal_to_2(list(jieba.cut(sentence, cut_all=True)))
            transcript = self.g2pm(sentence, tone=False)
            transcript = " ".join(transcript)
            # 使用列表推导式替换所有的 "nu"
            transcript = transcript.replace('u:', 'v')
            transcript = transcript.split()
            waveform, sample_rate = torchaudio.load(wav_file)
            waveform = waveform[0:1]
            emission, token_spans = self.compute_alignments(waveform, transcript)
            num_frames = emission.size(1)
            ratio = waveform.size(1) / num_frames
            for word in seg_list:
                result = find_all_occurrences(sentence, word)
                for i, (s, e) in enumerate(result):
                    start = s
                    end = e - 1
                    x0 = int(ratio * token_spans[start][0].start)
                    x1 = int(ratio * token_spans[end][-1].end)
                    score = np.mean([_score(token_spans[i]) for i in range(start, end+1)])
                    if score > 0.3: # 设置了平均阈值
                        save_path = os.path.join(self.save_dirs, word) + "/" + "_".join([wav_file.split('/')[-3], wav_file.split('/')[-2], wav_file.split('/')[-1][:-4], str(i)]) + '.wav'
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torchaudio.save(save_path, waveform[:, x0:x1], sample_rate=sample_rate)
        except Exception as e:
            print(wav_file)
            print(e)

                    

def data_process(
    sub_files, 
    device_id
):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_id}'
        model = TORCHAUDIO_MFA(device_id='0')  # 使用不同的设备ID
        for wav_file in tqdm(sub_files):
            txt_file = wav_file.replace('.wav', '_norm.txt')
            sentence = read_text_file(txt_file)
            model.make_mfa(wav_file, sentence)
        del model
    except Exception as e:
        print(e)



def multiprocess_to_(
    todo_list,
    num_gpu=[0],  # Default to GPU 0 if num_gpu is not provided
    num_process=3,
):
    num_available_gpus = len(num_gpu)
    with multiprocessing.Pool(processes=num_process) as pool:
        for i in range(num_process):
            sub_files = todo_list[i::num_process]
            device_id = num_gpu[i % num_available_gpus]
            pool.apply_async(data_process, args=(sub_files, device_id))
        pool.close()
        pool.join()
    
    
if __name__ == "__main__":
    target_dirs = "/server24/aizq/wenetspeech_clips/M_S" # S: 151600 & M_S: 1362900
    save_dirs = "/server24/aizq/mm_kws/datasets/WenetPhrase/M_S"
    todo_list = find_files_with_suffix(target_dirs, '.wav')
    print(len(todo_list))
    num_gpu=[0, 1, 2, 3, 4, 5]
    multiprocess_to_(
        todo_list,
        num_gpu=num_gpu,
        num_process=len(num_gpu) * 5,
    )