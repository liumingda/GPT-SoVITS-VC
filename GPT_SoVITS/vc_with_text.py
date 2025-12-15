import json
import logging
import os
import re
import sys
import traceback
import warnings

import torch
import torchaudio
from text.LangSegmenter import LangSegmenter

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)

version = model_version = os.environ.get("version", "v2")

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import change_choices, get_weights_names, name2gpt_path, name2sovits_path

SoVITS_names, GPT_names = get_weights_names()
from config import pretrained_sovits_name

path_sovits_v3 = pretrained_sovits_name["v3"]
path_sovits_v4 = pretrained_sovits_name["v4"]
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)

if os.path.exists("./weight.json"):
    pass
else:
    with open("./weight.json", "w", encoding="utf-8") as file:
        json.dump({"GPT": {}, "SoVITS": {}}, file)

with open("./weight.json", "r", encoding="utf-8") as file:
    weight_data = file.read()
    weight_data = json.loads(weight_data)
    sovits_path = os.environ.get("sovits_path", weight_data.get("SoVITS", {}).get(version, SoVITS_names[0]))
    if isinstance(sovits_path, list):
        sovits_path = sovits_path[0]

cnhubert_base_path = os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
infer_ttswebui = os.environ.get("infer_ttswebui", 9873)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

import gradio as gr
import librosa
import numpy as np
from feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer

cnhubert.cnhubert_base_path = cnhubert_base_path

import random
from GPT_SoVITS.module.models import Generator, SynthesizerTrn, SynthesizerTrnV3

def set_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000000)
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

from time import time as ttime
from peft import LoraConfig, get_peft_model
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
v3v4set = {"v3", "v4"}

def change_sovits_weights(sovits_path):
    if "！" in sovits_path or "!" in sovits_path:
        sovits_path = name2sovits_path[sovits_path]
    global vq_model, hps, version, model_version, if_lora_v3
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    print(sovits_path, version, model_version, if_lora_v3)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    if if_lora_v3 == True and is_exist == False:
        info = path_sovits + "SoVITS %s" % model_version + i18n("底模缺失，无法加载相应 LoRA 权重")
        gr.Warning(info)
        raise FileExistsError(info)
    
    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version

    if model_version not in v3v4set:
        if "Pro" not in model_version:
            model_version = version
        else:
            hps.model.version = model_version
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else:
        hps.model.version = model_version
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    
    if if_lora_v3 == False:
        print("loading sovits_%s" % model_version, vq_model.load_state_dict(dict_s2["weight"], strict=False))
    else:
        path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        print(
            "loading sovits_%spretrained_G" % model_version,
            vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False),
        )
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        print("loading sovits_%s_lora%s" % (model_version, lora_rank))
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        vq_model.eval()

    # 更新 weight.json
    with open("./weight.json") as f:
        data = f.read()
        data = json.loads(data)
        data["SoVITS"][version] = sovits_path
    with open("./weight.json", "w") as f:
        f.write(json.dumps(data))
    
    # return i18n("模型加载成功:") + sovits_path
    return f'''
    <div style="
        border: 1px solid #e5e7eb;
        border-left: 5px solid #4ade80;
        background-color: #f6fdf9;
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-top: 10px;
        color: #166534;
    ">
        <div style="font-size: 14px; font-weight: bold;">
            ✅ {i18n("模型加载成功")}
        </div>
        <div style="font-size: 12px; margin-top: 4px; opacity: 0.8; word-break: break-all;">
            {sovits_path}
        </div>
    </div>
    '''

try:
    change_sovits_weights(sovits_path)
except:
    pass

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
now_dir = os.getcwd()

def clean_hifigan_model():
    global hifigan_model
    if hifigan_model:
        hifigan_model = hifigan_model.cpu()
        hifigan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass

def clean_bigvgan_model():
    global bigvgan_model
    if bigvgan_model:
        bigvgan_model = bigvgan_model.cpu()
        bigvgan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass

def clean_sv_cn_model():
    global sv_cn_model
    if sv_cn_model:
        sv_cn_model.embedding_model = sv_cn_model.embedding_model.cpu()
        sv_cn_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass

def init_bigvgan():
    global bigvgan_model, hifigan_model, sv_cn_model
    from BigVGAN import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=False,
    )
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    clean_hifigan_model()
    clean_sv_cn_model()
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)

def init_hifigan():
    global hifigan_model, bigvgan_model, sv_cn_model
    hifigan_model = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0,
        is_bias=True,
    )
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load(
        "%s/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,),
        map_location="cpu",
        weights_only=False,
    )
    print("loading vocoder", hifigan_model.load_state_dict(state_dict_g))
    clean_bigvgan_model()
    clean_sv_cn_model()
    if is_half == True:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)

from sv import SV

def init_sv_cn():
    global hifigan_model, bigvgan_model, sv_cn_model
    sv_cn_model = SV(device, is_half)
    clean_bigvgan_model()
    clean_hifigan_model()

bigvgan_model = hifigan_model = sv_cn_model = None
if model_version == "v3":
    init_bigvgan()
if model_version == "v4":
    init_hifigan()
if model_version in {"v2Pro", "v2ProPlus"}:
    init_sv_cn()

resample_transform_dict = {}

def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)

def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = resample(audio, sr0, sr1, device)
    else:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)

    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro == True:
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio

def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

dtype = torch.float16 if is_half == True else torch.float32

def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)
    return bert

from text import chinese

def get_phones_and_bert(text, language, version, final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "all_zh":
            if re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext, "zh", version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "all_yue" and re.search(r"[A-Za-z]", formattext):
            formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
            formattext = chinese.mix_text_normalize(formattext)
            return get_phones_and_bert(formattext, "yue", version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist = []
        langlist = []
        if language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(dtype), norm_text

from module.mel_processing import mel_spectrogram_torch, spectrogram_torch

spec_min = -12
spec_max = 2

cache = {}

@torch.no_grad()
def vc_main(
    ref_wav_path,
    source_wav_path,
    source_text,  
    speed=1,
    pitch_change=0, 
):
    global cache
    if ref_wav_path:
        pass
    else:
        gr.Warning(i18n("请上传参考音频"))

    if not source_wav_path:
        gr.Warning(i18n("请上传源音频"))
        
    if not source_text:
        gr.Warning(i18n("请填写源音频对应的文本"))

    if model_version in v3v4set:
        ref_free = False
    
    if model_version not in {"v3", "v4", "v2Pro", "v2ProPlus"}:
        clean_bigvgan_model()
        clean_hifigan_model()
        clean_sv_cn_model()
    t = []
    t0 = ttime()

    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        if is_half == True:
            wav16k = wav16k.half().to(device)
        else:
            wav16k = wav16k.to(device)

        vc_wav16k, sr = librosa.load(source_wav_path, sr=16000)

        vc_wav16k = torch.from_numpy(vc_wav16k)
        if is_half == True:
            vc_wav16k = vc_wav16k.half().to(device)
        else:
            vc_wav16k = vc_wav16k.to(device)
        
       
        vc_ssl_content = ssl_model.model(vc_wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        vc_codes = vq_model.extract_latent(vc_ssl_content)
        vc_prompt_semantic = vc_codes[0, 0]
        vc_prompt = vc_prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1 - t0)

    audio_opt = []
    
    t2 = ttime()
    t3 = ttime()
    is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
    
    if model_version not in v3v4set:
        refers = []
        if is_v2pro:
            sv_emb = []
            if sv_cn_model == None:
                init_sv_cn()
        
        if len(refers) == 0:
            refers, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device, is_v2pro)
            refers = [refers]
            if is_v2pro:
                sv_emb = [sv_cn_model.compute_embedding3(audio_tensor)]

        text = source_text
        print(f"Processing manual text: {text}")

        phones2, bert2, norm_text2 = get_phones_and_bert(text, "all_zh", version)
        
        if is_v2pro:
            # v2ProPlus
            audio = vq_model.decode(
                vc_prompt.unsqueeze(0), torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed, sv_emb=sv_emb
            )[0][0]
        else:
            # v2
            audio = vq_model.decode(
                vc_prompt.unsqueeze(0), torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed
            )[0][0]
   
    max_audio = torch.abs(audio).max()
    if max_audio > 1:
        audio = audio / max_audio
    audio_opt.append(audio)
    t4 = ttime()
    t.extend([t2 - t1, t3 - t2, t4 - t3])
    t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
    audio_opt = torch.cat(audio_opt, 0)
    if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    else:
        opt_sr = 48000  # v4
    
    audio_opt = audio_opt.cpu().detach().numpy()

    if pitch_change != 0:
        print(f"Applying pitch shift: {pitch_change} semitones")
        audio_opt = librosa.effects.pitch_shift(audio_opt, sr=opt_sr, n_steps=pitch_change)

    yield opt_sr, (audio_opt * 32767).astype(np.int16)


def custom_sort_key(s):
    parts = re.split("(\d+)", s)
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def html_center(text, label="p"):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


with gr.Blocks(title="GPT-SoVITS VC WebUI", analytics_enabled=False) as app:
    gr.HTML(
        top_html.format(
            i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.")
            + i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
        ),
        elem_classes="markdown",
    )

    gr.Markdown(html_center(i18n("GPT-SoVITS-VC-with-text"), "h2"))

    with gr.Row():
        SoVITS_dropdown = gr.Dropdown(label=i18n("SoVITS模型列表"), choices=sorted(SoVITS_names, key=custom_sort_key), value=SoVITS_names[0])
        refresh_button = gr.Button(i18n("刷新模型路径"))
        status_label = gr.Markdown(value="")
    
    with gr.Group() as vc_mode:
        gr.Markdown(html_center(i18n("语音转换：请上传参考音频，源音频并填写对应文本"), "h3"))
        with gr.Row():
            ref_wav = gr.Audio(label=i18n("参考音频（目标音色）"), type="filepath")
            with gr.Column():
                source_wav = gr.Audio(label=i18n("源音频（语义内容）"), type="filepath")
                source_text_input = gr.Textbox(
                    label=i18n("源音频对应的文本"), 
                    placeholder=i18n("请输入源音频中说的话"),
                    lines=2
                )
        
        with gr.Row():
            speed_slider = gr.Slider(minimum=0.6, maximum=1.65, step=0.05, label=i18n("语速调整"), value=1)
            pitch_shift_slider = gr.Slider(minimum=-12, maximum=12, step=1, label=i18n("变调 (半音)"), value=0)
        
        vc_button = gr.Button(i18n("执行语音转换"), variant="primary")
        vc_output = gr.Audio(label=i18n("输出结果"))
        
        vc_button.click(
            vc_main, 
            inputs=[ref_wav, source_wav, source_text_input, speed_slider, pitch_shift_slider], 
            outputs=[vc_output]
        )

    def refresh_models():
        SoVITS_names, GPT_names = get_weights_names()
        return gr.update(choices=sorted(SoVITS_names, key=custom_sort_key), value=SoVITS_names[0] if SoVITS_names else "")

    refresh_button.click(fn=refresh_models, inputs=[], outputs=[SoVITS_dropdown])

    SoVITS_dropdown.change(
        change_sovits_weights,
        [SoVITS_dropdown],
        [status_label]
    )

if __name__ == "__main__":
    app.queue().launch(server_name="0.0.0.0", inbrowser=True, share=is_share, server_port=infer_ttswebui)