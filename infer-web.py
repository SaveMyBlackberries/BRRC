import os
import logging
# 设置用于记录用户信息的日志记录器
log_path = os.path.abspath('user_logs/usage_logs.txt')
user_logger = logging.getLogger('user_logger')
user_logger.setLevel(logging.INFO)

# 创建文件处理器
user_file_handler = logging.FileHandler(log_path)
user_file_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(message)s')
user_file_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
user_logger.addHandler(user_file_handler)

# 禁止日志记录器向其他地方传播
user_logger.propagate = False
import sys
from dotenv import load_dotenv
import requests
import wave
import tempfile
import zipfile
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import UVRHANDLER
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from i18n.i18n import I18nAuto
from configs.config import Config
from sklearn.cluster import MiniBatchKMeans
import torch
import numpy as np
import gradio as gr
import faiss
import fairseq
import librosa
import librosa.display
import pathlib
import chardet
import re
import pandas as pd
from datetime import datetime
import json
from pydub import AudioSegment
import lameenc
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import shutil

import matplotlib.pyplot as plt
import soundfile as sf
from dotenv import load_dotenv
from tools import pretrain_helper

import edge_tts, asyncio
from infer.modules.vc.ilariatts import tts_order_voice
language_dict = tts_order_voice
ilariavoices = list(language_dict.keys())

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % now_dir, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "models/pth"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

config = Config()
vc = VC(config)

weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))

uvr5_names = [
    '5_HP-Karaoke-UVR.pth',
    'Kim_Vocal_2.onnx',
    'MDX23C-8KFFT-InstVoc_HQ_2.ckpt',
    'UVR-DeEcho-DeReverb.pth',
    'UVR-Denoise',
]
if config.dml:
    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res


    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
i18n = I18nAuto()
logger.info(i18n)
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
                value in gpu_name.upper()
                for value in [
                    "10",
                    "16",
                    "20",
                    "30",
                    "40",
                    "A2",
                    "A3",
                    "A4",
                    "P4",
                    "A50",
                    "500",
                    "A60",
                    "70",
                    "80",
                    "90",
                    "M4",
                    "T4",
                    "TITAN",
                ]
        ):
            if_gpu_ok = True
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = ((min(mem) // 2 + 1) // 2) * 2
else:
    gpu_info = i18n("Your GPU doesn't work for training")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"

weight_root = os.getenv("weight_root")
index_root = os.getenv("index_root")
audio_root = "audios"
sup_audioext = {'wav', 'mp3', 'flac', 'ogg', 'opus',
                'm4a', 'mp4', 'aac', 'alac', 'wma',
                'aiff', 'webm', 'ac3'}

names        = [os.path.join(root, file)
               for root, _, files in os.walk(weight_root)
               for file in files
               if file.endswith((".pth", ".onnx"))]

indexes_list = [os.path.join(root, name)
               for root, _, files in os.walk(index_root, topdown=False) 
               for name in files 
               if name.endswith(".index") and "trained" not in name]
audio_paths  = [os.path.join(root, name)
               for root, _, files in os.walk(audio_root, topdown=False) 
               for name in files
               if name.endswith(tuple(sup_audioext))]
def get_pretrained_files(directory, keyword, filter_str):
    file_paths = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pth") and keyword in filename and filter_str in filename:
            file_paths[filename] = os.path.join(directory, filename)
    return file_paths

pretrained_directory = "assets/pretrained_v2"
pretrained_path = {filename: os.path.join(pretrained_directory, filename) for filename in os.listdir(pretrained_directory)}
pretrained_G_files = get_pretrained_files(pretrained_directory, "G", "f0")
pretrained_D_files = get_pretrained_files(pretrained_directory, "D", "f0")

def get_pretrained_models(path_str, f0_str, sr2):
    sr_mapping = pretrain_helper.get_pretrained_models(f0_str)

    pretrained_G_filename = sr_mapping.get(sr2, "")
    pretrained_D_filename = pretrained_G_filename.replace("G", "D")

    if not pretrained_G_filename or not pretrained_D_filename:
        logging.warning(f"Pretrained models not found for sample rate {sr2}, will not use pretrained models")

    return os.path.join(pretrained_directory, pretrained_G_filename), os.path.join(pretrained_directory, pretrained_D_filename)

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))

def generate_spectrogram_and_get_info(audio_file):
    y, sr = librosa.load(audio_file, sr=None)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    log_S = librosa.amplitude_to_db(S, ref=np.max, top_db=256)

    plt.figure(figsize=(12, 5.5))
    librosa.display.specshow(log_S, sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f dB', pad=0.01)
    plt.tight_layout(pad=0.5)

    plt.savefig('spectrogram.png', dpi=500)

    audio_info = sf.info(audio_file)
    bit_depth = {'PCM_16': 16, 'FLOAT': 32}.get(audio_info.subtype, 0)
    minutes, seconds = divmod(audio_info.duration, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds *= 1000
    speed_in_kbps = audio_info.samplerate * bit_depth / 1000
    filename_without_extension, _ = os.path.splitext(os.path.basename(audio_file))

    info_table = f"""
    | Information | Value |
    | :---: | :---: |
    | File Name | {filename_without_extension} |
    | Duration | {int(minutes)} minutes - {int(seconds)} seconds - {int(milliseconds)} milliseconds |
    | Bitrate | {speed_in_kbps} kbp/s |
    | Audio Channels | {audio_info.channels} |
    | Samples per second | {audio_info.samplerate} Hz |
    | Bit per second | {audio_info.samplerate * audio_info.channels * bit_depth} bit/s |
    """

    return info_table, "spectrogram.png"


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    audios = [os.path.join(audio_root, file) for file in os.listdir(os.path.join(now_dir, "audios"))]

    return {"choices": sorted(names), "__type__": "update"}, {"choices": sorted(index_paths),"__type__": "update"},{
        "choices": sorted(audios), "__type__": "update"
    }

#定义用于更新图片的函数
def update_image(voice):
    image_dict = {
        "01皮皮第2版.pth": "img/01pipi.png",
        "01皮皮第3版.pth": "img/01pipi.png",
        "02蛋君v2.pth": "img/02dj.png",
        "02蛋君v3.pth": "img/02dj.png",
        "03蛋妹1.pth": "img/03dm.png",
        "04成年温柔女性，素养思维课专用旁白.pth": "img/04nv.png",
        "05莉莉.pth": "img/05lili.png",
        "06科科小男生.pth": "img/06keke.png",
        "07佳佳.pth": "img/07jiajia.png",
        "08乐乐小男孩.pth": "img/08lele.png",
        "09甜甜.pth": "img/09tiantian.png",
        "10小轩胖男孩.pth": "img/10xiaoxuan.png",
        "11小安.pth": "img/11xiaoan.png",
        "12年轻活泼女老师.pth": "img/12fmteacher.png",
        "13年轻男老师.pth": "img/13mteacher.png",
        "14小孩音色.pth": "img/14kid.png",
        "15成年男声.pth": "img/15male.png",
        "16朵朵.pth": "img/16duoduo.png",
        "17老年男人.pth": "img/17oldman.png",
        "18礼花蛋龙套A阳光开朗男人.pth": "img/18longtao.png",
        "19小易.pth": "img/19xiaoyi.png",
        "20小猴的皮皮,学龄前的调皮男孩.pth": "img/20tipy.png",
        "50----以下歌手未获授权，请勿商用----.pth": "img/50mei.png",
        "51迈克杰克逊.pth": "img/51mj.png",
        "52ArianaGrande.pth": "img/52ag.png",
        "53DojaCat.pth": "img/53doja.png",
        "54Adele.pth": "img/54adele.png",
        "55施瓦辛格.pth": "img/55schwazenge.png",
        "BP-jennie-all-round.pth": "img/bp-jennie.png",
        "BP-jenniesoft.pth": "img/bp-jennie.png",
        "BP-jenniestrong.pth": "img/bp-jennie.png",
        "BP-jisoo-all-round.pth": "img/bp-jisoo.png",
        "BP-jisoosoft.pth": "img/bp-jisoo.png",
        "BP-jisoostrong.pth": "img/bp-jisoo.png",
        "BP-lisa-all-round.pth": "img/bp-lisa.png",
        "BP-lisasweet.pth": "img/bp-lisa.png",
        "BP-rosesoft.pth": "img/bp-rose.png",
        "BP-rosestrong.pth": "img/bp-rose.png",
        "BP-soft-lisa.pth": "img/bp-rose.png",
        "ILLIT-Iroha.pth": "img/ILLIT-iroha.png",
        "ILLIT-Minju.pth": "img/ILLIT-Minju.png",
        "ILLIT-Moka.pth": "img/ILLIT-moka.png",
        "ILLIT-Wonhee.pth": "img/ILLIT-Wonhee.png",
        "ILLIT-Yunah.pth": "img/ILLIT-Yunah.png",
        "ITZY-Chaeryeong.pth": "img/ITZY-chaeryeong.png",
        "ITZY-Lia1200.pth": "img/ITZY-lia.png",
        "ITZY-LiaDeepSoft.pth": "img/ITZY-lia.png",
        "ITZY-Ryujin2.pth": "img/ITZY-ryujin.png",
        "ITZY-Ryujin300.pth": "img/ITZY-ryujin.png",
        "ITZY-Yeji.pth": "img/ITZY-yeji.png",
        "ITZY-Yeji500.pth": "img/ITZY-yeji.png",
        "ITZY-Yuna.pth": "img/ITZY-yuna.png",
        # 添加其他音色和图片的对应关系
    }
    image_path = image_dict.get(voice, "img/default.png")
    return gr.update(value=image_path)

#这个函数用于生成指定时长的静音音频
def generate_silence(duration_ms, sample_rate=24000, bit_depth=16):
    num_frames = int(sample_rate * duration_ms / 1000)
    silent_frame = b'\x00' * (bit_depth // 8) * num_frames

    encoder = lameenc.Encoder()
    encoder.set_channels(1)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_bit_rate(128)
    encoder.set_out_sample_rate(sample_rate)
    encoder.set_quality(2)

    mp3_data = encoder.encode(silent_frame)
    mp3_data += encoder.flush()

    return mp3_data


async def run_tts(text, voice, rate, volume):
    segments = re.split(r'(\<\!p\d+\>)', text)
    combined_audio = b''
    try:
        for segment in segments:
            if re.match(r'\<\!p\d+\>', segment):
                pause_duration = int(re.search(r'\d+', segment).group())
                silence = await asyncio.to_thread(generate_silence, pause_duration)
                combined_audio += silence
            else:
                communicate = edge_tts.Communicate(segment, voice, rate=rate, volume=volume)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        combined_audio += chunk["data"]
                #combined_audio += combined_audio

        with open("./TEMP/temp_ilariatts.mp3", "wb") as f:
            f.write(combined_audio)
        return "./TEMP/temp_ilariatts.mp3"
    except Exception as e:
        raise RuntimeError(f"Error processing TTS: {e}")
        
#记录变声的时长信息
def vc_log(
    spk_item,
    input_audio0,
    input_audio1,
    vc_transform0,
    f0_file,
    f0method0,
    file_index1,
    file_index2,
    index_rate1,
    filter_radius0,
    resample_sr0,
    rms_mix_rate0,
    protect0,
    user_name,
):
    is_valid, error_message, user_department = validate_name_department(user_name)
    if not is_valid:
        return error_message, None, gr.update(visible=False), gr.update(visible=False)   # 返回错误信息
    # 调用 vc_single 并获取结果
    output1, output2, audio_length = vc.vc_single(
        spk_item,
        input_audio0,
        input_audio1,
        vc_transform0,
        f0_file,
        f0method0,
        file_index1,
        file_index2,
        index_rate1,
        filter_radius0,
        resample_sr0,
        rms_mix_rate0,
        protect0,
    )

    # 记录日志
    user_logger.info(f"User: {user_name}, Department: {user_department}, Converted Audio Length: {audio_length} seconds")
    # 返回结果
    return output1, output2

def parse_usage_logs(file_path):
    #with open(file_path, 'rb') as f:
    #    result = chardet.detect(f.read())
    #encoding = result['encoding']
    #print(encoding)
    with open(file_path, 'r', encoding='gb18030') as f: # 使用检测到的编码读取文件
        logs = f.readlines()
    
    data = []
    for log in logs:
        match_audio = re.match(
            r'(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - User: (?P<user>.*?), Department: (?P<department>.*?), Converted Audio Length: (?P<length>[\d.]+) seconds',
            log.strip()
        )
        match_tts = re.match(
            r'(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - User: (?P<user>.*?), Department: (?P<department>.*?), TTS Characters: (?P<characters>\d+)',
            log.strip()
        )
        if match_audio:
            log_data = match_audio.groupdict()
            log_data['datetime'] = datetime.strptime(log_data['datetime'], '%Y-%m-%d %H:%M:%S,%f')
            log_data['length'] = float(log_data['length'])
            log_data['characters'] = 0
            data.append(log_data)
        elif match_tts:
            log_data = match_tts.groupdict()
            log_data['datetime'] = datetime.strptime(log_data['datetime'], '%Y-%m-%d %H:%M:%S,%f')
            log_data['length'] = 0.0
            log_data['characters'] = int(log_data['characters'])
            data.append(log_data)
    
    df = pd.DataFrame(data)
    return df

def generate_report():
    try:
        df = parse_usage_logs('user_logs/usage_logs.txt')
        
        # 过滤本月的数据
        current_month = datetime.now().strftime('%Y-%m')
        df = df[df['datetime'].dt.strftime('%Y-%m') == current_month]
        
        # 按姓名和部门进行分组并汇总
        report = df.groupby(['user', 'department'], as_index=False).agg({
            'length': 'sum',
            'characters': 'sum'
        })
        report.rename(columns={
            'user': '-----姓名-----',
            'department': '------部门-----',
            'length': '本月用量（音频时长/秒）',
            'characters': '本月用量（TTS字符数）'
        }, inplace=True)
        
        # 排序
        report = report.sort_values(by='本月用量（TTS字符数）', ascending=False)
        
        # 转换为HTML表格
        return report.to_html(index=False, header=True, table_id="usage_report_table")
    
    except Exception as e:
        return f"Error generating report: {str(e)}"
        
def validate_name_department(name):
    #with open('user_logs/usage_logs.txt', 'rb') as f:
    #    result = chardet.detect(f.read())
    #encoding = result['encoding']
    #print(encoding)
    with open('user_logs/usage_logs.txt', 'r', encoding='gb18030') as f:
        logs = f.readlines()
    #添加用户特别命令解释
    if name.startswith("~?~"):
        try:
            # 解析用户名称和部门名称
            parts = name[3:].split(',')
            if len(parts) == 2:
                user_name = parts[0].strip()
                department_name = parts[1].strip()

                # 添加新日志行到文件
                with open('user_logs/usage_logs.txt', 'a', encoding='gb18030') as f:
                    new_log = f"2024-01-01 13:13:52,811 - User: {user_name}, Department: {department_name}, TTS Characters: 0\n"
                    f.write(new_log)
            #else:
                #return False, "请输入正确的中文姓名", None
        except Exception as e:
            return False, f"添加记录时发生错误: {e}", None
    
    
    if not re.match(r'^[\u4e00-\u9fa5]{2,10}$', name):
        return False, "请输入正确的中文姓名", None
    # 读取 usage_logs.txt 文件
    #with open('user_logs/usage_logs.txt', 'r', encoding=encoding) as f:
        #logs = f.readlines()

    # 使用正则表达式匹配用户信息
    for log in logs:
        match = re.match(
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - User: (.*?), Department: (.*?),', log.strip()
        )
        
        if match and match.group(1) == name:
            return True, "", match.group(2)  # 找到匹配的姓名后，返回对应的部门信息

    # 如果未找到匹配的记录
    return False, "您暂未获得使用权限，请知音楼联系管理员开通：工号082826", None


# Define the tts_and_convert function
def tts_and_convert(ttsvoice, text, rate, spk_item, vc_transform, f0_file, f0method, file_index1, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, user_name):
    rate_map = {
        "快速": "+20%",
        "原速": "+0%",
        "稍慢": "-20%",
        "较慢": "-50%"
    }
    rate_value = rate_map.get(rate, "+0%")
    volume = "+0%"  # 不改变音量
    pitch = "+0Hz"  # 不改变音高
    # Validate name
    is_valid, error_message, user_department = validate_name_department(user_name)
    if not is_valid:
        return error_message, None, gr.update(visible=False), gr.update(visible=False)   # 返回错误信息
    # 记录字符数
    char_count = len(text)

    # Perform TTS (we only need 1 function)
    vo=language_dict[ttsvoice]
    #asyncio.run(edge_tts.Communicate(text, vo, rate=rate_value, volume=volume, pitch=pitch).save("./TEMP/temp_ilariatts.mp3"))
    aud_path = asyncio.run(run_tts(vo, rate_value, volume))
    #Calls vc similar to any other inference.
    #This is why we needed all the other shit in our call, otherwise we couldn't infer.
    output1, output2, audio_length = vc.vc_single(spk_item , None,aud_path, vc_transform, f0_file, f0method, file_index1, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect)
    
    # 记录音频长度
    #audio_length = librosa.get_duration(filename=output2)
    
    

    # 记录日志信息
    user_logger.info(f"User: {user_name}, Department: {user_department}, TTS Characters: {char_count}")
    return output1, output2, gr.update(visible=True), gr.update(visible=True)

def convert_to_mp3(audio_file):
    if not audio_file:
        return None
    mp3_path = audio_file.replace(".wav", ".mp3")
    audio_segment = AudioSegment.from_wav(audio_file)
    audio_segment.export(mp3_path, format="mp3", parameters=["-q:a", "0"])
    return mp3_path

def handle_download(audio_info, audio_data, ilariatext):
    if not audio_data:
        return None
    
    ## 在这里可以使用 ilariatext 进行处理
    prefix = ilariatext[:4]+"_" if ilariatext else "abc"
    
    # 假设返回的是 (sample_rate, audio_data)
    sample_rate, audio_data = audio_data
    # 创建一个临时的wav文件
    with tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=".wav") as tmp_wav:
        sf.write(tmp_wav.name, audio_data, sample_rate)
        tmp_wav_path = tmp_wav.name
        
    # 转换为mp3
    mp3_path = convert_to_mp3(tmp_wav_path)
    return mp3_path

def import_files(file):
    if file is not None:
        file_name = file.name
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(file.name, 'r') as zip_ref:
                # Create a temporary directory to extract files
                temp_dir = './TEMP'
                zip_ref.extractall(temp_dir)
                # Move .pth and .index files to their respective directories
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.pth'):
                            destination = './models/pth/' + file
                            if not os.path.exists(destination):
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(f"File {destination} already exists. Skipping.")
                        elif file.endswith('.index'):
                            destination = './models/index/' + file
                            if not os.path.exists(destination):
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(f"File {destination} already exists. Skipping.")
                # Remove the temporary directory
                shutil.rmtree(temp_dir)
            return "Zip file has been successfully extracted."
        elif file_name.endswith('.pth'):
            destination = './models/pth/' + os.path.basename(file.name)
            if not os.path.exists(destination):
                os.rename(file.name, destination)
            else:
                print(f"File {destination} already exists. Skipping.")
            return "PTH file has been successfully imported."
        elif file_name.endswith('.index'):
            destination = './models/index/' + os.path.basename(file.name)
            if not os.path.exists(destination):
                os.rename(file.name, destination)
            else:
                print(f"File {destination} already exists. Skipping.")
            return "Index file has been successfully imported."
        else:
            return "Unsupported file type."
    else:
        return "No file has been uploaded."

def import_button_click(file):
    return import_files(file)

def calculate_remaining_time(epochs, seconds_per_epoch):
    total_seconds = epochs * seconds_per_epoch

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours == 0:
        return f"{int(minutes)} minutes"
    elif hours == 1:
        return f"{int(hours)} hour and {int(minutes)} minutes"
    else:
        return f"{int(hours)} hours and {int(minutes)} minutes"

def get_audio_duration(audio_file_path):
    audio_info = sf.info(audio_file_path)
    duration_minutes = audio_info.duration / 60
    return duration_minutes
      
def clean():
    return {"value": "", "__type__": "update"}


sr_dict = {
    "32k": 32000, "40k": 40000, "48k": 48000, "OV2-32k": 32000, "OV2-40k": 40000, "RIN-40k": 40000, "Snowie-40k": 40000, "Snowie-48k": 48000, "SnowieV3.1-40k": 40000, "SnowieV3.1-32k": 32000, "SnowieV3.1-48k": 48000, "SnowieV3.1-RinE3-40K": 40000, "Italia-32k": 32000,
}

def durations(sample_rate, model_options, qualities, duration):
    if duration <= 350:
        return qualities['short']
    else:
        if sample_rate == 32000:
            return model_options['32k']
        elif sample_rate == 40000:
            return model_options['40k']
        elif sample_rate == 48000:
            return model_options['48k']
        else:
            return qualities['other']

def get_training_info(audio_file):
    if audio_file is None:
        return 'Please provide an audio file!'
    duration = get_audio_duration(audio_file)
    sample_rate = wave.open(audio_file, 'rb').getframerate()

    training_info = {
        (0, 2): (150, 'OV2'),
        (2, 3): (200, 'OV2'),
        (3, 5): (250, 'OV2'),
        (5, 10): (300, 'Normal'),
        (10, 25): (500, 'Normal'),
        (25, 45): (700, 'Normal'),
        (45, 60): (1000, 'Normal')
    }

    for (min_duration, max_duration), (epochs, pretrain) in training_info.items():
        if min_duration <= duration < max_duration:
            break
    else:
        return 'Duration is not within the specified range!'

    return f'You should use the **{pretrain}** pretrain with **{epochs}** epochs at **{sample_rate/1000}khz** sample rate.'


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True

def on_button_click(audio_file_path):
    return get_training_info(audio_file_path)

def download_from_url(url, model):
    if url == '':
        return "URL cannot be left empty."
    if model == '':
        return "You need to name your model. For example: Ilaria"

    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)

    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)

    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile

    try:
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        else:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            with open(zipfile_path, 'wb') as file:
                file.write(response.content)

        shutil.unpack_archive(zipfile_path, "./unzips", 'zip')

        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.makedirs(f'./models/index', exist_ok=True)
                    shutil.copy2(file_path, f'./models/index/{model}.index')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    os.makedirs(f'./models/pth', exist_ok=True)
                    shutil.copy(file_path, f'./models/pth/{model}.pth')

        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "Model downloaded, you can go back to the inference page!"

    except subprocess.CalledProcessError as e:
        return f"ERROR - Download failed (gdown): {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"ERROR - Download failed (requests): {str(e)}"
    except Exception as e:
        return f"ERROR - The test failed: {str(e)}"

def transfer_files(filething, dataset_dir='dataset/'):
    file_names = [f.name for f in filething]
    for f in file_names:
        filename = os.path.basename(f)
        destination = os.path.join(dataset_dir, filename)
        shutil.copyfile(f, destination)
    return "Transferred files to dataset directory!"

def if_done_multi(done, ps):
    while 1:
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    logger.info(cmd)
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield f.read()
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                    '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                    % (
                        config.python_cmd,
                        now_dir,
                        exp_dir,
                        n_p,
                        f0method,
                    )
            )
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                            '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                            % (
                                config.python_cmd,
                                leng,
                                idx,
                                n_g,
                                now_dir,
                                exp_dir,
                                config.is_half,
                            )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )
                    ps.append(p)
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                        config.python_cmd
                        + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                        % (
                            now_dir,
                            exp_dir,
                        )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )
                p.wait()
                done = [True]
        while 1:
            with open(
                    "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield f.read()
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log

    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
                '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'
                % (
                    config.python_cmd,
                    config.device,
                    leng,
                    idx,
                    n_g,
                    now_dir,
                    exp_dir,
                    version19,
                )
        )
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )
        ps.append(p)
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield f.read()
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log



def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["32k","40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["32k", "40k", "48k", "OV2-32k", "OV2-40k", "RIN-40k","Snowie-40k","Snowie-48k","Italia-32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )

def change_f0(if_f0_3, sr2, version19):
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 is True else "", sr2),
    )

def click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
):
    global f0_dir, f0nsf_dir
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
    feature_dir = (
        "%s/3_feature256" % exp_dir
        if version19 == "v1"
        else "%s/3_feature768" % exp_dir
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % exp_dir
        f0nsf_dir = "%s/2b-f0nsf" % exp_dir
        names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy"
                "|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
                '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s '
                "-sw %s -v %s"
                % (
                    config.python_cmd,
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    gpus16,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
        )
    else:
        cmd = (
                '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw '
                "%s -v %s"
                % (
                    config.python_cmd,
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
        )
    logger.info(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "You can view console or train.log"

def train_index(exp_dir1, version19):
    exp_dir = "logs/%s" % exp_dir1
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % exp_dir
        if version19 == "v1"
        else "%s/3_feature768" % exp_dir
    )
    if not os.path.exists(feature_dir):
        return "Please perform Feature Extraction First!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please perform Feature Extraction First！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i: i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "Success，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    yield "\n".join(infos)

F0GPUVisible = config.dml is False

def change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}

vc_output1 = gr.Textbox(label=i18n("状态信息"))
vc_output2 = gr.Audio(label=i18n("试听"))


with gr.Blocks(title="未来之声▶︎ •၊၊||၊|။||||။‌‌‌‌‌၊|• 3:00", show_api=False) as app:
    gr.Markdown("<h1> 未来之声-语音模型3.0▶︎ •၊၊||၊|။||||။‌‌‌‌‌၊|• 3:00</h1>")
    gr.Markdown(value=i18n(" 📼本软件使用开源RVC搭建📼 "))
    gr.Markdown(i18n("所有声音模型仅限于学而思素养使用，未经授权禁止转发或用于其他用途"))
    
    with gr.Tabs():
        with gr.TabItem(i18n("AI配音转换")):
            with gr.Row():
                with gr.Column():
                    sid0= gr.Dropdown(label=i18n("角色选择"), choices=sorted(names), value="01皮皮第2版.pth")
                    vc.get_vc("01皮皮第2版.pth", None, None)
                    #user_department = gr.Textbox(label="部门", placeholder="请输入你的部门/分校名")
                    user_name = gr.Textbox(label="姓名", placeholder="请输入你的姓名")
                    gr.Markdown('''
                        <p style='color:grey; font-size:11px; font-style:italic;'>
                        因本工具仅限公司内部使用，故请填写您的实际名字
                        </p>
                    ''')
                # 添加图片
                image_display = gr.Image(value="img/01pipi.png", shape=(40, 40), show_label=False)
                sid0.change(
                    fn=update_image,
                    inputs=[sid0],
                    outputs=image_display
                )
                sid1= sid0
                
                with gr.Column():
                    refresh_button = gr.Button(i18n("刷新音色"), variant="secondary")
                    clean_button = gr.Button(i18n("清空音色"), variant="tertiary")
                vc_transform0 = gr.inputs.Slider(
                                label=i18n(
                                    "变调: -24 表示降低2个八度 (男) 24 表示升高2个八度 (女)"),
                                minimum=-24,
                                maximum=24,
                                default=0,
                                step=1,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], #api_name="infer_clean"
                )

            
            with gr.TabItem(i18n("音色转换")):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():                                
                                input_audio1 = gr.Audio(
                                    label=i18n("拖入音频文件"),
                                    type="filepath",
                                )
                                #input_audio0 = ''
                                ##record_button = gr.Audio(source="microphone", label="麦克风",
                                ##                         type="filepath")
                                ##input_audio0 = ''
                                input_audio0 = gr.Dropdown(
                                    label=i18n("Select a file from the audio folder"),
                                    choices=sorted(audio_paths),
                                    value='',
                                    interactive=True,
                                    visible=False
                                )
                                ##record_button.change(
                                ##    fn=lambda x: x,
                                ##    inputs=[record_button],
                                ##    outputs=[input_audio0],
                                ##)
                                but0 = gr.Button(i18n("立即变声"))
                                file_index1 = gr.Textbox(
                                    label=i18n("指定索引文件"),
                                    placeholder=".\models\index",
                                    interactive=True,
                                    visible=False,
                                )
                                vc_output1.render()
                                vc_output2.render()
                                file_index2 = gr.Textbox(
                                    label=i18n("选择列出的索引文件"),
                                    choices=sorted(index_paths),
                                    interactive=True,
                                    visible=False,
                                )
                        with gr.Column():
                            with gr.Accordion('Advanced Settings', open=False, visible=False):
                                with gr.Column():
                                    f0method0 = gr.Radio(
                                        label=i18n("Pitch Extraction"),
                                        choices=["harvest", "crepe"]
                                        if config.dml is False
                                        else ["harvest", "rmvpe"],
                                        value="harvest",
                                        interactive=True,
                                    )
                                    resample_sr0 = gr.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=i18n("Resampling, 0=none"),
                                        value=0,
                                        step=1,
                                        interactive=True,
                                    )
                                    rms_mix_rate0 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("0=Input source volume, 1=Normalized Output"),
                                        value=0.25,
                                        interactive=True,
                                    )
                                    protect0 = gr.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=i18n(
                                            "Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open"),
                                        value=0.33,
                                        step=0.01,
                                        interactive=True,
                                    )
                                    filter_radius0 = gr.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=i18n(">=3 apply median filter to the harvested pitch results"),
                                        value=3,
                                        step=1,
                                        interactive=True,
                                    )
                                    index_rate1 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("Index Ratio"),
                                        value=0.40,
                                        interactive=True,
                                    )
                                    f0_file = gr.File(
                                        label=i18n("F0 curve file [optional]"),
                                        visible=False,
                                    )

                                    refresh_button.click(
                                        fn=change_choices,
                                        inputs=[],
                                        outputs=[sid0, file_index2, input_audio1],
                                        #api_name="infer_refresh",
                                    )
                                    file_index1 = gr.Textbox(
                                        label=i18n("Path of index"),
                                        placeholder="%userprofile%\\Desktop\\models\\model_example.index",
                                        interactive=True,
                                    )
                                    file_index2 = gr.Dropdown(
                                        label=i18n("Auto-detect index path"),
                                        choices=sorted(index_paths),
                                        interactive=True,
                                    )
                                    spk_item = gr.Slider(
                                        minimum=0,
                                        maximum=2333,
                                        step=1,
                                        label=i18n("Speaker ID (Auto-Detected)"),
                                        value=0,
                                        visible=True,
                                        interactive=False,
                                    )

                            with gr.Accordion('文本生成语音', open=True):
                                with gr.Column():
                                    ilariaid=gr.Dropdown(label="选择方言", choices=ilariavoices, interactive=True, value="活泼女声")
                                    ilariatext = gr.Textbox(label="此处输入要生成的文本，用<!p1000>表示停顿1秒", interactive=True, value="教不好学生等于偷钱<!p1000>抢钱", lines=5)
                                    rate_selection = gr.Dropdown(
                                        label="选择语速",
                                        choices=["快速", "原速", "稍慢", "较慢"],
                                        value="原速",
                                        interactive=True
                                    )
                                    ilariatts_button = gr.Button(value="立即生成")
                                    download_button = gr.Button(value="点击下载MP3文件", visible=False)
                                    download_output = gr.File(label="点击下载生成的MP3文件", visible=False)
                                    
                                    
                                    ilariatts_button.click(tts_and_convert,
                                                           [ilariaid,
                                                            ilariatext,
                                                            rate_selection,
                                                            spk_item,
                                                            vc_transform0,
                                                            f0_file,
                                                            f0method0,
                                                            file_index1,
                                                            file_index2,
                                                            index_rate1,
                                                            filter_radius0,
                                                            resample_sr0,
                                                            rms_mix_rate0,
                                                            protect0,
                                                            user_name],
                                                           [vc_output1, vc_output2, download_button, download_output])
                                    #with gr.Row():
                                    #    vc_output1.render()
                                    #    vc_output2.render()

                                    # 点击下载按钮时的处理逻辑
                                    download_button.click(
                                        fn=handle_download, 
                                        inputs=[vc_output1, vc_output2, ilariatext], 
                                        outputs=[download_output])
                                    
                                    
                            
                                      #Otherwise everything break, to be optimized
                            with gr.Accordion('高级设置', visible=True, open=False):
                                with gr.Column():
                                    f0method0 = gr.Radio(
                                        label=i18n("Pitch Extraction以下设置适合高级玩家，一般情况下不要修改"),
                                        choices=["harvest", "crepe"]
                                        if config.dml is False
                                        else ["harvest", "rmvpe"],
                                        value="harvest",
                                        interactive=True,
                                    )
                                    resample_sr0 = gr.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=i18n("Resampling, 0=none"),
                                        value=0,
                                        step=1,
                                        interactive=True,
                                    )
                                    rms_mix_rate0 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("0=Input source volume, 1=Normalized Output"),
                                        value=0.25,
                                        interactive=True,
                                    )
                                    protect0 = gr.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=i18n(
                                            "Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open"),
                                        value=0.33,
                                        step=0.01,
                                        interactive=True,
                                    )
                                    filter_radius0 = gr.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=i18n(">=3 apply median filter to the harvested pitch results"),
                                        value=3,
                                        step=1,
                                        interactive=True,
                                    )
                                    index_rate1 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=i18n("Index Ratio"),
                                        value=0.40,
                                        interactive=True,
                                    )
                                    f0_file = gr.File(
                                        label=i18n("F0 curve file [optional]"),
                                        visible=False,
                                    )

                                    refresh_button.click(
                                        fn=change_choices,
                                        inputs=[],
                                        outputs=[sid0, file_index2],
                                        #api_name="infer_refresh",
                                    )
                                    file_index1 = gr.Textbox(
                                        label=i18n("Path of index"),
                                        placeholder="%userprofile%\\Desktop\\models\\model_example.index",
                                        interactive=True,
                                    )
                                    file_index2 = gr.Dropdown(
                                        label=i18n("Auto-detect index path"),
                                        choices=sorted(index_paths),
                                        interactive=True,
                                    )

                ##with gr.Group():
                    with gr.Column():
                        ##but0 = gr.Button(i18n("立即变声"), variant="primary")
                        #with gr.Row():
                        #    vc_output1.render()
                        #    vc_output2.render()

                        but0.click(
                            fn=vc_log,
                            inputs=[
                                spk_item,
                                input_audio0,
                                input_audio1,
                                vc_transform0,
                                f0_file,
                                f0method0,
                                file_index1,
                                file_index2,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                                user_name,
                            ],
                            outputs=[vc_output1, vc_output2],

                        )

            with gr.TabItem(i18n("批量转换")):
                gr.Markdown(
                    value=i18n("批量转换")
                )
                gr.Markdown(i18n("此功能目前仅保留给管理员使用，一般用户无法使用，请使用音色转换页"))
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n("变调，男转女需升高值，女转男需降低，最大24，最小-24"),
                            value=0
                        )
                        opt_input = gr.Textbox(label=i18n("输出路径"))
                        file_index3 = gr.Textbox(
                            label=i18n("Path to index"),
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n("Auto-detect index path"),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label=i18n("Pitch Extraction"),
                            choices=["harvest", "crepe"]
                            if config.dml is False
                            else ["harvest", "rmvpe"],
                            value="harvest",
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label=i18n("Export Format"),
                            choices=["flac", "wav", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )

                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            #api_name="infer_refresh_batch",
                        )

                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("Resampling, 0=none"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("0=Input source volume, 1=Normalized Output"),
                            value=0.25,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open"),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3 apply median filter to the harvested pitch results"),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Index Ratio"),
                            value=0.40,
                            interactive=True,
                        )
                with gr.Row():
                    dir_input = gr.Textbox(
                        label=i18n("Enter the path to the audio folder to be processed"),
                        placeholder="%userprofile%\\Desktop\\covers",
                        
                    )
                    inputs = gr.File(
                        file_count="multiple", label=i18n("Audio files can also be imported in batch")
                    )

                with gr.Row():
                    but1 = gr.Button(i18n("Convert"), variant="primary")
                    vc_output3 = gr.Textbox(label=i18n("Console"))

                    but1.click(
                        vc.vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            # file_big_npy2,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                        ],
                        [vc_output3],
                        #api_name="infer_convert_batch",
                    )
        with gr.TabItem(i18n("详情")):
                with gr.Accordion('模型信息', open=False):
                    with gr.Column():
                        sid1 = gr.Dropdown(label=i18n("音色模型"), choices=sorted(names))
                        refresh_button = gr.Button(i18n("刷新"), variant="primary")
                        refresh_button.click(
                         fn=change_choices,
                            inputs=[],
                            outputs=[sid1, file_index2],
                            #api_name="infer_refresh",
                            )
                        modelload_out = gr.Textbox(label="模型元数据", interactive=False, lines=4)
                        get_model_info_button = gr.Button(i18n("获取信息"))
                        get_model_info_button.click(
                         fn=vc.get_vc, 
                         inputs=[sid1, protect0, protect1],
                         outputs=[spk_item, protect0, protect1, file_index2, file_index4, modelload_out]
                        )
                    
                with gr.Accordion('本月用量统计', open=True):
                    report_button = gr.Button("生成统计报告")
                    report_output = gr.HTML()  # 用于显示生成的报告表格
        
                    report_button.click(
                        fn=generate_report, 
                        inputs=[], 
                        outputs=report_output
                    )

                with gr.Accordion('感谢', open=False):
                    gr.Markdown('''
                ## 感谢：
                
		### 项目发起者
		
		- 唐睿老师
		- 对新技术研发的耐心
		- 以及对工具落地的推动

		
		### 数据支持
        
		- 徐习习老师
		- 素养设计团队
		
		### 试用和反馈
		
		- 索菲菲老师
		- 思维小低教研团队的伙伴们
        
		
		### 配音流程支持
		
		- 牛海陇，姜钊泉老师
		- 感谢在技术初期不断协助进行试错
		- 在推广初期承担全部的配音工作
		- 为蛋君的发声探索了一套方法
		
		### 音频工程技术人员
		
		- 孙靖濛
		- 张博伦
		- 吴崧铭
                                
                ### 知音楼搜索 未来之声AI语音系统客服群 交流和提问
                ''')
                    
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[spk_item, protect0, protect1, file_index2, file_index4, modelload_out],
                    #api_name="infer_change_voice",
                )      
        with gr.TabItem(i18n("")):
            gr.Markdown('''
                ![ilaria](https://i.ytimg.com/vi/5PWqt2Wg-us/maxresdefault.jpg)
            ''')
    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )

