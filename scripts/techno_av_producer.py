#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
techno_av_producer.py

一体化脚本（在你现有版本基础上增强）：
1) 从本地样本库（Berlin Underground）加载 WAV，预处理成 log-mel 训练集
2) 训练一个轻量 VAE（conv2d encoder/decoder）学习 Drum/Bass/Synth/Pad 等 Loop 的声学分布
3) 采样生成若干新 Loop（Griffin-Lim 反谱重建）
4) 按 A/V 情绪规则 + 采样库自动编曲并导出 WAV

数据根（请确认目录真实存在）：
~/Documents/ualproject/data/dataset/techno dataset/Berlin underground/
  ├── Bass Loop
  ├── Claps
  ├── Drum Loop
  ├── Hats
  ├── Kick
  ├── Pad Loop
  ├── Percussion
  ├── Rides
  ├── SFX
  ├── Snares
  ├── Synth Loop
  └── Toms
"""

import os, sys, math, glob, random, json, time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===================== 路径与超参 =====================

DATA_ROOT = os.path.expanduser(
    "~/Documents/ualproject/data/dataset/techno dataset/Berlin underground"
)

SAVE_ROOT   = os.path.abspath("./techno_workdir")
GEN_LOOP_DIR= os.path.join(SAVE_ROOT, "generated_loops")
MODEL_DIR   = os.path.join(SAVE_ROOT, "models")
os.makedirs(GEN_LOOP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,    exist_ok=True)

# 参与学习/生成的类（与你现有一致）
CATEGORIES_TO_LEARN = ["Drum Loop", "Bass Loop", "Synth Loop", "Pad Loop"]
# 仅用于编曲铺垫的打击类
CATEGORIES_TO_USE_EXTRA = ["Kick", "Claps", "Hats", "Percussion", "Rides", "SFX", "Snares", "Toms"]

# 音频与特征
SR = 44100
TARGET_SEC = 4.0
N_FFT = 1024
HOP   = 256
N_MELS= 128
GRIFFIN_ITER = 96   # ↑ 更清晰

# 训练
BATCH  = 16
EPOCHS = 100
LR     = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 生成
LOOPS_PER_CLASS = 6
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 编曲
BPM = 128
MASTER_SEC = 60
PEAK_LIM_DB = -1.0
MASTER_GAIN_DB = -1.5

# ===================== 实用函数 =====================

def list_wavs(root, subfolder):
    p = os.path.join(root, subfolder)
    if not os.path.isdir(p): return []
    files = []
    for ext in ("*.wav","*.aif","*.aiff","*.flac","*.mp3"):
        files.extend(glob.glob(os.path.join(p, ext)))
    return sorted(files)

def load_mono(path, sr=SR, target_sec=TARGET_SEC):
    y, _ = librosa.load(path, sr=sr, mono=True)
    tgt_len = int(sr * target_sec)
    if len(y) < tgt_len:
        reps = int(np.ceil(tgt_len / max(1, len(y))))
        y = np.tile(y, reps)[:tgt_len]
    elif len(y) > tgt_len:
        start = np.random.randint(0, len(y)-tgt_len+1)
        y = y[start:start+tgt_len]
    return y.astype(np.float32)

def wav_to_logmel(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, power=2.0)
    S = np.maximum(S, 1e-10)
    logS = librosa.power_to_db(S, ref=np.max)       # -80..0 dB
    logS_norm = (logS + 80.0) / 80.0                # → 0..1
    return logS_norm.astype(np.float32)             # [n_mels, T]

def logmel_to_wav(logS_norm):
    logS = (logS_norm * 80.0) - 80.0
    S = librosa.db_to_power(logS)
    y = librosa.feature.inverse.mel_to_audio(S, sr=SR, n_fft=N_FFT,
                                             hop_length=HOP, n_iter=GRIFFIN_ITER)
    return y.astype(np.float32)

def db_to_gain(db):
    return 10.0 ** (db / 20.0)

def limiter(x, peak_db=PEAK_LIM_DB):
    peak = float(np.max(np.abs(x) + 1e-9))
    tgt = db_to_gain(peak_db)
    if peak > tgt:
        x = x * (tgt / peak)
    return x

# ===================== 数据集（增强多样性） =====================

class LoopDataset(Dataset):
    def __init__(self, root, categories, augment=True):
        self.files = []
        for c in categories:
            self.files += list_wavs(root, c)
        if len(self.files) == 0:
            raise RuntimeError(f"未在 {root} 下找到 {categories} 的音频文件。")
        self.augment = augment

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        y = load_mono(p)

        if self.augment:
            # 随机增益
            y *= np.random.uniform(0.85, 1.15)

            # 微小变速（显式 rate=，兼容不同 librosa 版本）
            if np.random.rand() < 0.6:
                rate = float(np.random.uniform(0.92, 1.08))
                y = librosa.effects.time_stretch(y, rate=rate)
                tgt = int(SR * TARGET_SEC)
                if len(y) < tgt:
                    reps = int(np.ceil(tgt / max(1, len(y))))
                    y = np.tile(y, reps)[:tgt]
                else:
                    y = y[:tgt]

            # 轻度移调（对带音高类更有用）
            if np.random.rand() < 0.5 and any(k in p for k in ["Bass", "Synth", "Pad"]):
                semis = float(np.random.uniform(-2.0, 2.0))
                y = librosa.effects.pitch_shift(y, sr=SR, n_steps=semis)

            # 去点击
            if np.random.rand() < 0.2:
                k = np.random.randint(1, 3)
                for _ in range(k):
                    i = np.random.randint(0, len(y)-16)
                    y[i:i+16] *= np.linspace(1, 0, 16, endpoint=False)

        logmel = wav_to_logmel(y)                 # [n_mels, T]
        x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, T]
        return x

# ===================== 轻量 VAE =====================

class ConvVAE(nn.Module):
    def __init__(self, in_ch=1, z_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, 2, 1),    nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128,3, 2, 1),    nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256,3, 2, 1),   nn.LeakyReLU(0.2, True),
        )
        # 动态推断特征尺寸
        dummy = torch.zeros(1, 1, N_MELS, int(TARGET_SEC*SR/HOP))
        with torch.no_grad():
            z_feat = self.enc(dummy)
        c,h,w = z_feat.shape[1:]
        self.feat_shape = (c,h,w)
        flat = c*h*w

        self.fc_mu  = nn.Linear(flat, 64)
        self.fc_log = nn.Linear(flat, 64)

        self.fc_dec = nn.Linear(64, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128,64, 4,2,1), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 4,2,1), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 16, 4,2,1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 1, 3,1,1), nn.Sigmoid()    # 输出 0..1
        )

    def encode(self, x):
        z = self.enc(x)
        z = z.view(z.size(0), -1)
        mu = self.fc_mu(z)
        logvar = self.fc_log(z)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), *self.feat_shape)
        x = self.dec(h)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar

# --------- 对齐补丁（关键修复） ---------
def _align_like(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    将 pred(N,C,F,T) 在频率维(F)和时间维(T)对齐到 tgt 的尺寸：
    - pred 更大：裁到 tgt
    - pred 更小：末尾 0 填充
    """
    assert pred.dim()==4 and tgt.dim()==4
    _, _, Fp, Tp = pred.shape
    _, _, Ft, Tt = tgt.shape

    # 频率维
    if Fp > Ft:
        pred = pred[:, :, :Ft, :]
    elif Fp < Ft:
        pred = F.pad(pred, pad=(0,0,0, Ft-Fp))  # pad=(T_left,T_right,F_top,F_bottom)

    # 时间维
    _, _, Fp2, Tp = pred.shape
    if Tp > Tt:
        pred = pred[:, :, :, :Tt]
    elif Tp < Tt:
        pred = F.pad(pred, pad=(0, Tt-Tp, 0, 0))

    return pred

def vae_loss(x, x_rec, mu, logvar, beta=1e-3):
    x_rec = _align_like(x_rec, x)
    recon = F.l1_loss(x_rec, x)
    kld   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta*kld, recon.item(), kld.item()

# ===================== 训练 & 生成 =====================

def train_vae():
    ds = LoopDataset(DATA_ROOT, CATEGORIES_TO_LEARN, augment=True)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    model = ConvVAE(z_dim=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"[TRAIN] files={len(ds)}  steps/epoch≈{len(dl)}  device={DEVICE}")
    best = 1e9
    for ep in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {ep}/{EPOCHS}")
        losses = []
        for x in pbar:
            x = x.to(DEVICE)
            x_rec, mu, logvar = model(x)
            loss, rec, kld = vae_loss(x, x_rec, mu, logvar, beta=1e-3)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}", rec=f"{rec:.4f}", kld=f"{kld:.4f}")
        avg = float(np.mean(losses))
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "loop_vae.pt"))
            print(f"[SAVE] best loss={best:.4f}")
    return os.path.join(MODEL_DIR, "loop_vae.pt")

def gen_loops(model_ckpt):
    model = ConvVAE(z_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(model_ckpt, map_location=DEVICE))
    model.eval()

    # —— 多均值策略：对四个类别分别编码若干真实样本，提取多个“簇中心”以增强多样性 ——
    real_by_cat = {c: list_wavs(DATA_ROOT, c) for c in CATEGORIES_TO_LEARN}
    cat_means = {c: [] for c in CATEGORIES_TO_LEARN}

    with torch.no_grad():
        for c in CATEGORIES_TO_LEARN:
            paths = real_by_cat[c]
            if not paths: continue
            sel = random.sample(paths, min(64, len(paths)))
            # 分成若干组，组内求均值当作“簇中心”
            group = 8
            chunk = max(1, len(sel)//group)
            for i in range(0, len(sel), chunk):
                part = sel[i:i+chunk]
                Z = []
                for p in part:
                    y = load_mono(p)
                    x = torch.tensor(wav_to_logmel(y), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    mu, _ = model.encode(x)
                    Z.append(mu.squeeze(0).cpu().numpy())
                if Z:
                    cat_means[c].append(np.mean(np.stack(Z,0),0))

    out_files = []
    for c in CATEGORIES_TO_LEARN:
        means = cat_means.get(c, [])
        if not means:  # 兜底：全局 0
            means = [np.zeros(64, dtype=np.float32)]
        for i in range(LOOPS_PER_CLASS):
            m  = means[i % len(means)]
            std= 0.6 + 0.6*np.random.rand()   # 温度范围
            z  = m + np.random.randn(64).astype(np.float32) * std
            with torch.no_grad():
                zt = torch.tensor(z, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                x_rec = model.decode(zt).cpu().numpy()[0,0]
            y = logmel_to_wav(x_rec)
            y = limiter(y, peak_db=PEAK_LIM_DB)
            fade = int(0.01*SR)
            y[:fade] *= np.linspace(0,1,fade)
            y[-fade:] *= np.linspace(1,0,fade)
            fn = os.path.join(GEN_LOOP_DIR, f"{c.replace(' ','_')}_gen_{i+1:02d}.wav")
            sf.write(fn, y, SR)
            out_files.append(fn)

    print(f"[GEN] 生成 loops：{len(out_files)} → {GEN_LOOP_DIR}")
    return out_files

# ===================== 情绪驱动编曲 =====================

@dataclass
class Section:
    name: str
    bars: int
    density: float
    av_bias: float

def bars_to_samples(bars, bpm=BPM, sr=SR):
    sec_per_bar = 60.0 / bpm * 4.0
    return int(sec_per_bar * bars * sr)

def fit_loop_to_bars(loop, bars, bpm=BPM, sr=SR):
    """把任意 loop（数组）拉伸/裁切到整数小节长度。"""
    tgt_len = bars_to_samples(bars, bpm, sr)
    # librosa 的 time_stretch 输出时长 ≈ 输入长度 / rate
    # 想得到目标长度 → rate = len(loop)/tgt_len
    rate = float(len(loop)) / float(max(1, tgt_len))
    y = librosa.effects.time_stretch(loop, rate=rate)
    if len(y) < tgt_len:
        reps = int(np.ceil(tgt_len / max(1, len(y))))
        y = np.tile(y, reps)[:tgt_len]
    else:
        y = y[:tgt_len]
    return y.astype(np.float32)

def gain_db(y, db):
    return y * db_to_gain(db)

def mix_to_length(tracks, length):
    if len(tracks)==0:
        return np.zeros(length, dtype=np.float32)
    out = np.zeros(length, dtype=np.float32)
    for t in tracks:
        L = min(length, len(t))
        out[:L] += t[:L]
    out = np.tanh(out * 1.5)
    out = limiter(out, PEAK_LIM_DB)
    return out

def pick_files(cat, max_pick=8):
    files = list_wavs(DATA_ROOT, cat) + glob.glob(os.path.join(GEN_LOOP_DIR, f"{cat.replace(' ','_')}_gen_*.wav"))
    random.shuffle(files)
    return files[:max_pick]

def av_to_arrangement(av_curve, total_bars=64):
    def normA(a):
        if np.min(a) < 0: return (a+1)/2
        return a
    A = normA(np.array([x[0] for x in av_curve], dtype=float))
    V = normA(np.array([x[1] for x in av_curve], dtype=float))
    A_med = float(np.median(A))

    plan = [
        Section("Intro",  8, 0.30 + 0.2*A_med, -0.1),
        Section("Build", 16, 0.55 + 0.3*A_med, +0.1),
        Section("Drop",  16, 0.85 + 0.1*A_med, +0.2),
        Section("Break",  8, 0.40 + 0.1*A_med, -0.05),
        Section("Drop2", 16, 0.90, +0.25),
    ]
    total = sum(s.bars for s in plan)
    if total != total_bars:
        scale = total_bars/total
        for s in plan:
            s.bars = max(4, int(round(s.bars*scale)))
    return plan

def render_track(av_curve, out_path="av_techno_demo.wav"):
    random.seed(SEED); np.random.seed(SEED)

    plan = av_to_arrangement(av_curve, total_bars=int(MASTER_SEC/(60/BPM*4)))
    total_len = bars_to_samples(sum(s.bars for s in plan))
    print("[ARR] sections:", " | ".join([f"{s.name}:{s.bars}" for s in plan]))

    kicks = pick_files("Kick", 6)
    claps = pick_files("Claps", 6)
    hats  = pick_files("Hats",  8)
    drums = pick_files("Drum Loop", 10)
    bass  = pick_files("Bass Loop",  8)
    syns  = pick_files("Synth Loop", 10)
    pads  = pick_files("Pad Loop",   8)
    fxes  = pick_files("SFX", 6) + pick_files("Percussion", 6) + pick_files("Rides", 4)

    tracks = []
    cur = 0

    for s in plan:
        seg_len = bars_to_samples(s.bars)
        seg_tracks = []

        # 取中位 A/V（如需实时驱动可改为逐段传入）
        A = float(np.median([x[0] for x in av_curve]));  A = (A+1)/2 if A<0 else A
        V = float(np.median([x[1] for x in av_curve]));  V = (V+1)/2 if V<0 else V
        density = np.clip(s.density, 0.0, 1.0)

        # Kick
        if kicks:
            k = random.choice(kicks)
            y,_ = librosa.load(k, sr=SR, mono=True)
            kick_bar = fit_loop_to_bars(y, s.bars)
            seg_tracks.append(gain_db(kick_bar, -6.0 + 4.0*density))

        # Drum
        if drums and density>0.35:
            d = random.choice(drums)
            y,_ = librosa.load(d, sr=SR, mono=True)
            y = fit_loop_to_bars(y, s.bars)
            seg_tracks.append(gain_db(y, -8.0 + 3.0*density))

        # Clap
        if claps and (s.name in ["Build","Drop","Drop2"]):
            c = random.choice(claps)
            y,_ = librosa.load(c, sr=SR, mono=True)
            y = fit_loop_to_bars(y, s.bars)
            seg_tracks.append(gain_db(y, -10.0 + 3.0*density))

        # Hat
        if hats and (density>0.4 or V>0.5):
            h = random.choice(hats)
            y,_ = librosa.load(h, sr=SR, mono=True)
            y = fit_loop_to_bars(y, s.bars)
            seg_tracks.append(gain_db(y, -12.0 + 6.0*V + 3.0*density))

        # Bass
        if bass and (density>0.45 or "Drop" in s.name):
            b = random.choice(bass)
            y,_ = librosa.load(b, sr=SR, mono=True)
            y = fit_loop_to_bars(y, s.bars)
            seg_tracks.append(gain_db(y, -10.0 + 4.0*(density + s.av_bias)))

        # Synth / Pad
        if "Break" in s.name and pads:
            p = random.choice(pads)
            y,_ = librosa.load(p, sr=SR, mono=True)
            y = fit_loop_to_bars(y, s.bars)
            seg_tracks.append(gain_db(y, -14.0 + 6.0*V))
        elif syns and (density>0.5 or "Drop" in s.name):
            sy = random.choice(syns)
            y,_ = librosa.load(sy, sr=SR, mono=True)
            y = fit_loop_to_bars(y, s.bars)
            seg_tracks.append(gain_db(y, -14.0 + 6.0*V + 3.0*density))

        # FX / 过门
        if fxes and s.name in ["Build","Drop","Drop2"] and random.random() < 0.6:
            f = random.choice(fxes)
            y,_ = librosa.load(f, sr=SR, mono=True)
            y = fit_loop_to_bars(y, min(4, s.bars))
            pad = np.zeros(bars_to_samples(max(0, s.bars-4)), dtype=np.float32)
            y = np.concatenate([y, pad], 0)
            seg_tracks.append(gain_db(y, -16.0 + 6.0*A))

        # 段落淡入淡出
        seg = mix_to_length(seg_tracks, seg_len)
        fade = int(0.02*SR)
        seg[:fade] *= np.linspace(0,1,fade)
        seg[-fade:] *= np.linspace(1,0,fade)

        tracks.append((cur, seg))
        cur += seg_len

    master = np.zeros(total_len, dtype=np.float32)
    for start, seg in tracks:
        L = min(len(seg), total_len-start)
        master[start:start+L] += seg[:L]

    master = limiter(master, PEAK_LIM_DB)
    master = master * db_to_gain(MASTER_GAIN_DB)
    sf.write(out_path, master, SR)
    print(f"[RENDER] 导出：{out_path}  长度：{len(master)/SR:.1f}s   BPM={BPM}")

# ===================== 一键跑通 =====================

def main():
    print("=== Techno A/V Producer ===")
    print(f"[DATA] root = {DATA_ROOT}")
    any_file = False
    for c in (CATEGORIES_TO_LEARN + CATEGORIES_TO_USE_EXTRA):
        fs = list_wavs(DATA_ROOT, c)
        print(f"  - {c:12s}: {len(fs)} files")
        any_file = any_file or (len(fs)>0)
    if not any_file:
        print("[ERR] 未找到任何样本文件，请检查 DATA_ROOT 路径。")
        return

    ckpt = os.path.join(MODEL_DIR, "loop_vae.pt")
    if not os.path.isfile(ckpt):
        print("\n[STEP] 训练 VAE（EPOCHS 可在顶部调大/调小）")
        ckpt = train_vae()
    else:
        print("\n[STEP] 发现已有模型，跳过训练 →", ckpt)

    print("\n[STEP] 用 VAE 生成 loops …")
    gen_loops(ckpt)

    print("\n[STEP] 情绪驱动编曲 …")
    av_curve = [
        (0.30, 0.50), (0.35, 0.50),
        (0.50, 0.55), (0.65, 0.60),
        (0.85, 0.60), (0.90, 0.65),
        (0.40, 0.45),
        (0.80, 0.60), (0.85, 0.65), (0.90, 0.70)
    ]
    out_wav = os.path.join(SAVE_ROOT, "av_techno_demo.wav")
    render_track(av_curve, out_path=out_wav)
    print("\n✅ 全流程完成！你可以直接试听：", out_wav)

if __name__ == "__main__":
    main()
