import os
from multiprocessing import Pool
from tqdm import tqdm
import torchaudio
import json

def process_audio(aidx):
    path = os.path.join(data_dir, wenetspeech['audios'][aidx]['path'][:-5] + '.wav')
    waveform, sr = torchaudio.load(path)
    
    for seg in range(len(wenetspeech['audios'][aidx]['segments'])):
        sid = wenetspeech['audios'][aidx]['segments'][seg]['sid']
        begin_time = wenetspeech['audios'][aidx]['segments'][seg]['begin_time']
        end_time = wenetspeech['audios'][aidx]['segments'][seg]['end_time']
        subsets = wenetspeech['audios'][aidx]['segments'][seg]['subsets']
        text = wenetspeech['audios'][aidx]['segments'][seg]['text']
        
        if 'M' in subsets and 'S' not in subsets:
            save_path = os.path.join("/server24/aizq/wenetspeech_clips/M_S", path.split('/')[-3], path.split('/')[-2], sid) + '.wav'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if not os.path.exists(save_path):
                torchaudio.save(save_path, waveform[:, int(begin_time * sr):int(end_time * sr)], sample_rate=sr)
                
            save_path_label = os.path.join("/server24/aizq/wenetspeech_clips/M_S", path.split('/')[-3], path.split('/')[-2], sid) + '.txt'
            
            if not os.path.exists(save_path_label):
                with open(save_path_label, "w") as file:
                    file.write(text)
        
        if 'S' in subsets:
            save_path = os.path.join("/server24/aizq/wenetspeech_clips/S", path.split('/')[-3], path.split('/')[-2], sid) + '.wav'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if not os.path.exists(save_path):
                torchaudio.save(save_path, waveform[:, int(begin_time * sr):int(end_time * sr)], sample_rate=sr)
                
            save_path_label = os.path.join("/server24/aizq/wenetspeech_clips/S", path.split('/')[-3], path.split('/')[-2], sid) + '.txt'
            
            if not os.path.exists(save_path_label):
                with open(save_path_label, "w") as file:
                    file.write(text)


if __name__ == '__main__':
    print("读取 WenetSpeech.json")
    with open('/server24/aizq/wenetspeech_UNTAR/WenetSpeech.json', 'r') as f: wenetspeech = json.load(f)
    print("读取完成正在处理")
    data_dir = "/server24/aizq/wenetspeech_UNTAR"
    num_processes = 32
    with Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(process_audio, range(len(wenetspeech['audios']))), total=len(wenetspeech['audios'])):
            pass
