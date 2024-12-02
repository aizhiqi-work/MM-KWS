import os
from multiprocessing import Pool
from tqdm import tqdm
import json
import os

def find_files_with_suffix(root_folder, endwiths):
    matching_files = []  # 以后缀名为键，文件路径列表为值初始化字典
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            for suffix in endwiths:
                if file.endswith(suffix):
                    matching_files.append(os.path.join(root, file))
    return matching_files

import os
import multiprocessing
from tqdm import tqdm
import subprocess
def process_file(file):
    output_file = file.replace('.txt', '_norm.txt')
    subprocess.run(['python', 'data-processing/wenetspeech/cn_tn.py', file, output_file])

root_folder = '/path/to/wenetspeech_clips/'
endwiths = ['.txt']
files = find_files_with_suffix(root_folder, endwiths)
with multiprocessing.Pool(32) as pool:
    # 使用 tqdm 并行迭代文件列表，并显示进度条
    for _ in tqdm(pool.imap_unordered(process_file, files), total=len(files)):
        pass