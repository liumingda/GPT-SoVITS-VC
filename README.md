# GPT-SoVITS 语音转换 (VC)

本项目基于 GPT-SoVITS 实现zero-shot 语音转换（Voice Conversion, VC）功能。它能够将源音频中的语音转换为参考音频的音色，同时保留源音频的语音内容和韵律语调。

## 概述
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
https://github.com/user-attachments/assets/da8566b5-06fd-4615-b0bd-e47b47c8acbe
注意: 运行此脚本需要预先安装 funasr 库，并确保已下载对应的 ASR 模型。

## 2. 带手动文本输入的语音转换 (vc_with_text.py)
该脚本允许用户手动输入源音频对应的文本内容，而对于v3, v4版本还需要输入参考音频对应的文本内容。当 ASR 识别效果不佳（如背景噪音大），或音频中包含生僻词、专有名词时，使用此模式可获得更准确的转换结果。

### 使用方法

运行以下命令启动 WebUI：

```bash
python GPT_SoVITS/vc_with_text.py
```
https://github.com/user-attachments/assets/a81c8084-dd99-4886-af99-1deea9c260ff
### 支持的模型版本
本项目支持 GPT-SoVITS 的所有主要版本：
v1, v2, v3, v4, v2pro, v2proplus
