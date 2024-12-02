# MM-KWS: Multi-modal Prompts for Multilingual User-defined Keyword Spotting

The official implementations of "Multi-modal Prompts for Multilingual User-defined Keyword Spotting" (accepted by Interspeech 2024).

[Paper](https://www.isca-archive.org/interspeech_2024/ai24_interspeech.html)


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
4. MM-KWS [WenetPhrase-test.csv](): data-processing/wenetphrase_test.csv



