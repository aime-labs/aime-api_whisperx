<h1 align="center">WhisperX</h1>

<p align="center">
  <a href="https://github.com/m-bain/whisperX/stargazers">
    <img src="https://img.shields.io/github/stars/m-bain/whisperX.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/m-bain/whisperX/issues">
        <img src="https://img.shields.io/github/issues/m-bain/whisperx.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/m-bain/whisperX/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/m-bain/whisperX.svg"
             alt="GitHub license">
  </a>
  <a href="https://arxiv.org/abs/2303.00747">
        <img src="http://img.shields.io/badge/Arxiv-2303.00747-B31B1B.svg"
             alt="ArXiv paper">
  </a>
  <a href="https://twitter.com/intent/tweet?text=&url=https%3A%2F%2Fgithub.com%2Fm-bain%2FwhisperX">
  <img src="https://img.shields.io/twitter/url/https/github.com/m-bain/whisperX.svg?style=social" alt="Twitter">
  </a>      
</p>


<img width="1216" align="center" alt="whisperx-arch" src="figures/pipeline.png">


<!-- <p align="left">Whisper-Based Automatic Speech Recognition (ASR) with improved timestamp accuracy + quality via forced phoneme alignment and voice-activity based batching for fast inference.</p> -->


<!-- <h2 align="left", id="what-is-it">What is it 🔎</h2> -->


This repository provides fast automatic speech recognition (70x realtime with large-v2) with word-level timestamps and speaker diarization.

- ⚡️ Batched inference for 70x realtime transcription using whisper large-v2
- 🪶 [faster-whisper](https://github.com/guillaumekln/faster-whisper) backend, requires <8GB gpu memory for large-v2 with beam_size=5
- 🎯 Accurate word-level timestamps using wav2vec2 alignment
- 👯‍♂️ Multispeaker ASR using speaker diarization from [pyannote-audio](https://github.com/pyannote/pyannote-audio) (speaker ID labels) 
- 🗣️ VAD preprocessing, reduces hallucination & batching with no WER degradation



**Whisper** is an ASR model [developed by OpenAI](https://github.com/openai/whisper), trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI's whisper does not natively support batching.

**Phoneme-Based ASR** A suite of models finetuned to recognise the smallest unit of speech distinguishing one word from another, e.g. the element p in "tap". A popular example model is [wav2vec2.0](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).

**Forced Alignment** refers to the process by which orthographic transcriptions are aligned to audio recordings to automatically generate phone level segmentation.

**Voice Activity Detection (VAD)** is the detection of the presence or absence of human speech.

**Speaker Diarization** is the process of partitioning an audio stream containing human speech into homogeneous segments according to the identity of each speaker.

<h2 align="left", id="highlights">New🚨</h2>

- 1st place at [Ego4d transcription challenge](https://eval.ai/web/challenges/challenge-page/1637/leaderboard/3931/WER)  🏆
- _WhisperX_ accepted at INTERSPEECH 2023 
- v3 transcript segment-per-sentence: using nltk sent_tokenize for better subtitlting & better diarization
- v3 released, 70x speed-up open-sourced. Using batched whisper with [faster-whisper](https://github.com/guillaumekln/faster-whisper) backend!
- v2 released, code cleanup, imports whisper library VAD filtering is now turned on by default, as in the paper.
- Paper drop🎓👨‍🏫! Please see our [ArxiV preprint](https://arxiv.org/abs/2303.00747) for benchmarking and details of WhisperX. We also introduce more efficient batch inference resulting in large-v2 with *60-70x REAL TIME speed.

## Installation with [AIME MLC](https://github.com/aime-team/aime-ml-containers) ⚙️

Easy installation within an [AIME ML-Container](https://github.com/aime-team/aime-ml-containers).

### 1. Create Python3.10 environment

```
mlc create whisper_container Pytorch 2.1.0
```
```
mlc open whisper_container
```

### 2. Install this repo

```
git clone https://github.com/aime-labs/aime-api_whisperx.git
cd aime-api_whisperx
pip install -e .
```

### 2. Install these system dependencies

Install FFmpeg and Rust support:
```
sudo apt update && sudo apt install ffmpeg
```
```
pip install setuptools-rust
```


<h2 align="left" id="inference"> Running inference </h2>
### Running WhisperX as HTTP/HTTPS API with AIME API Server

To run  WhisperX as HTTP/HTTPS API with [AIME API Server](https://github.com/aime-team/aime-api-server) start following Python script through the command line:

```bash
python3 run_whisper_with_api_server.py --api_server <address of api server>
```

It will start  WhisperX as worker, waiting for job request through the AIME API Server.



<h2 align="left" id="example">Usage 💬 (command line)</h2>

### 1. Run a basic transcription on an audio file:
Run whisper on example segment (using default params, whisper small) add `--highlight_words True` to visualise word timings in the .srt file.
```
python -m whisperx audio_file.mp3
```

### 2. Advanced Options

For increased timestamp accuracy, at the cost of higher gpu mem, use bigger models (bigger alignment model not found to be that helpful, see paper) e.g.
  ```
  python -m whisperx audio_file.mp3 --model large-v3 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 32
  ```

- Specify language:
  ```
  python -m whisperx audio_file.mp3 --language en
  ```

- Choose device:
  ```
  python -m whisperx audio_file.mp3 --device cuda
  ```

- Define output format:
  ```
  python -m whisperx audio_file.mp3 --output-format srt
  ```


<h2 align="left" id="whisper-mod">Technical Details 👷‍♂️</h2>

For specific details on the batching and alignment, the effect of VAD, as well as the chosen alignment model, see the preprint [paper](https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf).

To reduce GPU memory requirements, try any of the following (2. & 3. can affect quality):
1.  reduce batch size, e.g. `--batch_size 4`
2. use a smaller ASR model `--model base`
3. Use lighter compute type `--compute_type int8`

Transcription differences from openai's whisper:
1. Transcription without timestamps. To enable single pass batching, whisper inference is performed `--without_timestamps True`, this ensures 1 forward pass per sample in the batch. However, this can cause discrepancies the default whisper output.
2. VAD-based segment transcription, unlike the buffered transcription of openai's. In Wthe WhisperX paper we show this reduces WER, and enables accurate batched inference
3.  `--condition_on_prev_text` is set to `False` by default (reduces hallucination)

<h2 align="left" id="limitations">Limitations ⚠️</h2>

- Transcript words which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned and therefore are not given a timing.
- Overlapping speech is not handled particularly well by whisper nor whisperx
- Diarization is far from perfect
- Language specific wav2vec2 model is needed


<h2 align="left" id="contribute">Contribute 🧑‍🏫</h2>

If you are multilingual, a major way you can contribute to this project is to find phoneme models on huggingface (or train your own) and test them on speech for the target language. If the results look good send a pull request and some examples showing its success.

Bug finding and pull requests are also highly appreciated to keep this project going, since it's already diverging from the original research scope.


<h2 align="left" id="acks">Acknowledgements 🙏</h2>

This is builds on [openAI's whisper](https://github.com/openai/whisper).
Borrows important alignment code from [PyTorch tutorial on forced alignment](https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html)
And uses the wonderful pyannote VAD / Diarization https://github.com/pyannote/pyannote-audio


Valuable VAD & Diarization Models from [pyannote audio][https://github.com/pyannote/pyannote-audio]

Great backend from [faster-whisper](https://github.com/guillaumekln/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2)



---

> **Note**  
> This documentation is tailored specifically for the AIME server environment. For general WhisperX usage, refer to the [original repository](https://github.com/m-bain/whisperX).

--- 
