# GPT-SoVITS-VC (zero-shot)

本项目基于 GPT-SoVITS 实现zero-shot 语音转换（Voice Conversion, VC）功能。它能够将源音频中的语音转换为参考音频的音色，同时保留源音频的语音内容和韵律语调。本项目支持 GPT-SoVITS 的所有版本：v1, v2, v3, v4, v2pro, v2proplus

## 概述
效仿Cosyvoice的VC实现的各个版本的GPT-SoVits的VC

Step 1: 移除GPT模块，使用 Hubert (ssl_model) + VQ-VAE (vq_model) 提取源音频的 semantic token；

Step 2: 将参考音频传入模型提取音色特征；

Step 3: 使用 ASR 模型自动识别源音频（以及 v3/v4 的参考音频）的内容(或者手动输入)，送入 Text Encoder，帮助转换后的音频合成。

环境配置和模型下载与原项目相同，本项目目前提供两个主要的语音转换脚本：

1.  **`vc.py`**: 基于 ASR的自动语音转换。
2.  **`vc_with_text.py`**: 支持手动输入文本的语音转换。

## 1. 自动语音转换 (`vc.py`)

该脚本利用 ASR 模型（Paraformer）自动识别源音频中的文本内容。适用于源音频发音清晰的常规场景。

### 使用方法

运行以下命令启动 WebUI：

```bash
python GPT_SoVITS/vc.py
```

https://github.com/user-attachments/assets/2beff96f-cd05-4d5a-ae58-c664c6f81c6e

注意: 运行此脚本需要预先安装 funasr 库，并确保已下载对应的 ASR 模型。

## 2. 带手动文本输入的语音转换 (vc_with_text.py)
该脚本允许用户手动输入源音频对应的文本内容，而对于v3, v4版本还需要输入参考音频对应的文本内容。当 ASR 识别效果不佳（如背景噪音大），或音频中包含生僻词、专有名词时，使用此模式可获得更准确的转换结果。

### 使用方法

运行以下命令启动 WebUI：

```bash
python GPT_SoVITS/vc_with_text.py
```
https://github.com/user-attachments/assets/a81c8084-dd99-4886-af99-1deea9c260ff

### 参考项目
https://github.com/RVC-Boss/GPT-SoVITS

https://github.com/FunAudioLLM/CosyVoice

https://github.com/huangxu1991/GPT-SoVITS-VC
