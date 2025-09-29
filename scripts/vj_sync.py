#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vj_sync.py  ——  纯 CPU · 轻量 2B I2V · 不依赖 librosa

功能：
- 读取你的音乐 ~/Documents/ualproject/outputs/techno_workdir/av_techno_demo.wav
- 按固定 BPM 分段（默认 128，每段 4 小节）
- 用 SDXL Turbo 生成每段首帧（步数很低，嫌慢可改成纯色图）
- 用 CogVideoX 2B I2V（社区版）从首帧生视频（每段最多 8 帧）
- 用 ffmpeg 循环补齐到目标时长 + 交叉淡入拼接 + 叠回音轨
"""

# -------------------- 纯 CPU 安全环境 --------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""         # 禁 GPU
os.environ["PYTORCH_SDP_ATTENTION"] = "math"    # 关 sdpa 快路径
os.environ["NUMBA_DISABLE_JIT"] = "1"           # 虽未用 librosa，以防万一
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Mac 也不走 MPS

# -------------------- 标准库/依赖 --------------------
import math
import subprocess, shutil, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from PIL import Image

import torch
import torch.nn.functional as F

# MPS 不支持 float64：把任何 torch.arange(..., dtype=float64) 改成 float32（CPU 也安全）
_torch_arange = torch.arange
def _arange_patch(*args, **kwargs):
    if kwargs.get("dtype") == torch.float64:
        kwargs["dtype"] = torch.float32
    return _torch_arange(*args, **kwargs)
torch.arange = _arange_patch

# （可选）手写 SDPA（CPU 下也稳定）
_orig_sdpa = F.scaled_dot_product_attention
def _safe_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    d = q.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    attn = (q @ k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn = attn + attn_mask
    if is_causal:
        Lq, Lk = q.size(-2), k.size(-2)
        mask = torch.full((Lq, Lk), float("-inf"), device=attn.device, dtype=attn.dtype)
        mask = torch.triu(mask, diagonal=1)
        attn = attn + mask
    attn = attn.softmax(dim=-1)
    return attn @ v
F.scaled_dot_product_attention = _safe_sdpa

# -------------------- 业务参数（可改） --------------------
AUDIO_PATH = os.path.expanduser(
    "~/Documents/ualproject/outputs/techno_workdir/av_techno_demo.wav"
)
OUT_DIR     = os.path.expanduser("~/Documents/ualproject/outputs/vj_out_cpu")

THEME       = "Futuristic techno nightclub visuals, neon beams, volumetric light, abstract geometry"
NEG_PROMPT  = "low quality, text, watermark, logo, artifacts, blurry"

# 轻量 I2V（社区 2B 版；避免 5B 重负担）
# 若你已将 5B 模型下到本地目录，也可把 MODEL_I2V 改成那个本地绝对路径。
MODEL_I2V   = "NimVideo/cogvideox-2b-img2vid"     # 推荐
# MODEL_I2V = "zai-org/CogVideoX-5b-I2V"          # 更重，CPU 很慢，不建议

MODEL_T2I   = "stabilityai/sdxl-turbo"

BPM             = 128
BEATS_PER_BAR   = 4
BARS_PER_SEG    = 4               # 先短一点，加快首次打通
FPS             = 6               # 低帧率先跑通
WIDTH, HEIGHT   = 448, 256        # 低分辨率先跑通
XFADE_SEC       = 0.4
SEED            = 42

# 每段真实生成帧数（剩余用 ffmpeg 循环补时长）
MAX_FRAMES_PER_SEG = 8
I2V_STEPS          = 12           # I2V 推理步数（CPU 情况下保守一些）

DEVICE = "cpu"
DTYPE  = torch.float32

# -------------------- 工具函数 --------------------
def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未检测到 ffmpeg，请先安装：brew install ffmpeg")

def segment_by_bpm(audio_path: str, bpm: float, bars_per_seg: int, beats_per_bar: int=4):
    """按固定 BPM 均分，避免 librosa/numba。"""
    y, sr = sf.read(audio_path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    dur = len(y) / sr
    sec_per_beat = 60.0 / float(bpm)
    sec_per_bar  = sec_per_beat * float(beats_per_bar)
    seg_len_sec  = sec_per_bar * float(bars_per_seg)
    t = 0.0
    segs: List[Tuple[float, float]] = []
    while t < dur - 1e-3:
        t1 = min(dur, t + seg_len_sec)
        segs.append((t, t1))
        t = t1
    return segs, dur

# -------------------- SDXL Turbo：首帧 --------------------
def load_t2i():
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_T2I, dtype=DTYPE)
    except Exception:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(MODEL_T2I, dtype=DTYPE)
    pipe.to(DEVICE)
    return pipe

def gen_image(pipe_t2i, prompt, seed, w=WIDTH, h=HEIGHT):
    # 如果嫌慢，可以直接：return Image.new("RGB", (w, h), (0,0,0))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    img = pipe_t2i(
        prompt=prompt, negative_prompt=NEG_PROMPT,
        width=w, height=h, num_inference_steps=6, guidance_scale=1.2,
        generator=g
    ).images[0]
    return img

# -------------------- CogVideoX I2V：图生视频 --------------------
def load_i2v():
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.models.attention_processor import AttnProcessor

    model_id = MODEL_I2V
    # 若你已经本地下载到一个目录（含 model_index.json 等），也支持传本地路径：
    if os.path.isdir(model_id):
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, dtype=DTYPE)
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, dtype=DTYPE)

    pipe.to(DEVICE)
    try: pipe.transformer.set_attn_processor(AttnProcessor())
    except Exception: pass
    try: pipe.enable_attention_slicing()
    except Exception: pass
    try:
        pipe.vae.enable_tiling(); pipe.vae.enable_slicing()
    except Exception: pass
    return pipe

def i2v_from_image(pipe_i2v, image: Image.Image, prompt: str, seconds: float, fps=FPS, out_mp4="seg.mp4", seed=SEED):
    from diffusers.utils import export_to_video

    want_frames = max(6, int(round(seconds * fps)))
    num_frames  = min(want_frames, MAX_FRAMES_PER_SEG)
    frames = pipe_i2v(
        prompt=prompt,
        image=image.convert("RGB"),
        num_frames=num_frames,
        num_inference_steps=I2V_STEPS,
        guidance_scale=4.5,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    ).frames[0]
    export_to_video(frames, out_mp4, fps=fps)

    # 若不足时长，循环补齐
    short = len(frames) / float(fps)
    if short + 1e-3 < seconds:
        out_mp4, target_frames = loop_to_duration(out_mp4, seconds, fps)
        return out_mp4, frames[-1], target_frames
    else:
        return out_mp4, frames[-1], len(frames)

def loop_to_duration(src_mp4: str, target_sec: float, fps: int):
    require_ffmpeg()
    out_mp4 = src_mp4.replace(".mp4", "_fill.mp4")
    cmd = [
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", src_mp4,
        "-t", f"{target_sec:.3f}",
        "-vf", f"fps={fps},scale={WIDTH}:{HEIGHT},format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-an", out_mp4
    ]
    subprocess.run(cmd, check=True)
    return out_mp4, int(round(target_sec * fps))

# -------------------- 拼接 + 叠音轨 --------------------
def xfade_concat_with_audio(clips: List[str], frame_counts: List[int], fps: int, audio_path: str, out_path: str, xfade: float):
    require_ffmpeg()
    durs = [fc / float(fps) for fc in frame_counts]
    inputs = []
    for c in clips: inputs += ["-i", c]
    inputs += ["-i", audio_path]

    filt = []
    for i in range(len(clips)):
        filt.append(f"[{i}:v]fps={fps},scale={WIDTH}:{HEIGHT},format=yuv420p[v{i}];")
    chain = "[v0]"
    acc = durs[0]
    for i in range(1, len(clips)):
        off = max(0.0, acc - xfade)
        out_lbl = f"[m{i}]"
        filt.append(f"{chain}[v{i}]xfade=transition=fade:duration={xfade}:offset={off}{out_lbl};")
        chain = out_lbl
        acc += durs[i] - xfade
    filt.append(f"{chain}[vo]")

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", "".join(filt),
        "-map", "[vo]", "-map", f"{len(clips)}:a",
        "-shortest",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        out_path
    ]
    print("FFmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# -------------------- 主流程 --------------------
def main():
    audio = AUDIO_PATH
    assert os.path.isfile(audio), f"未找到音频：{audio}"
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[AUDIO] {audio}")
    print(f"[CPU only] dtype={DTYPE}, res={WIDTH}x{HEIGHT}, fps={FPS}, frames/seg≤{MAX_FRAMES_PER_SEG}")

    segs, dur = segment_by_bpm(audio, BPM, BARS_PER_SEG, BEATS_PER_BAR)
    print(f"[BPM={BPM}] 分段 {len(segs)} 段，每段 {BARS_PER_SEG} 小节，总长≈{dur:.1f}s")

    # 降低 diffusers 日志噪声
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()

    # 管线
    pipe_t2i = load_t2i()
    pipe_i2v = load_i2v()

    clips, frame_counts = [], []
    next_start = None
    random.seed(SEED)

    for idx, (t0, t1) in enumerate(segs):
        seg_sec = max(2.0, t1 - t0)
        seg_prompt = f"{THEME}, cinematic, soft camera motion, techno VJ, no text"
        img0 = gen_image(pipe_t2i, seg_prompt, SEED + idx, w=WIDTH, h=HEIGHT) if next_start is None else next_start
        seg_mp4 = str(Path(OUT_DIR) / f"seg_{idx:02d}.mp4")
        print(f"[I2V] 段 {idx+1}/{len(segs)}  {seg_sec:.2f}s → {Path(seg_mp4).name}")
        seg_mp4, last_frame, n_frames = i2v_from_image(
            pipe_i2v, img0, seg_prompt, seg_sec, fps=FPS, out_mp4=seg_mp4, seed=SEED+idx
        )
        clips.append(seg_mp4); frame_counts.append(n_frames); next_start = last_frame

    final_mp4 = str(Path(OUT_DIR) / "vj_final.mp4")
    if len(clips) == 1:
        subprocess.run(["ffmpeg", "-y", "-i", clips[0], "-i", audio, "-c:v", "copy", "-c:a", "aac", "-shortest", final_mp4], check=True)
    else:
        xfade_concat_with_audio(clips, frame_counts, FPS, audio, final_mp4, XFADE_SEC)

    print(f"\n✅ 完成：{final_mp4}")
    print(f"   升到 24fps：ffmpeg -y -i {final_mp4} -vf \"minterpolate=fps=24\" -c:v libx264 -preset medium -crf 20 -c:a copy {Path(OUT_DIR)/'vj_final_24fps.mp4'}")

if __name__ == "__main__":
    main()
