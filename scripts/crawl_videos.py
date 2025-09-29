#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, glob, shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- yt-dlp ---
try:
    from yt_dlp import YoutubeDL
    try:
        # 新版在 utils 里提供了 match_filter_func
        from yt_dlp.utils import match_filter_func as _match_filter_func
        HAS_MATCH_FILTER = True
    except Exception:
        HAS_MATCH_FILTER = False
except Exception as e:
    print("[ERR] 需要安装 yt-dlp：  pip install -U yt-dlp")
    raise

import cv2

# ---------- 可改参数 ----------
BASE_DIR = Path("/Users/chenyuxin/Documents/ualproject/data")
RAW_DIR = BASE_DIR / "raw_videos"
CLEAN_DIR = BASE_DIR / "clean_videos"
DATASET_DIR = BASE_DIR / "dataset" / "videos"

KEYWORDS = [
    "techno crowd",
    "techno audience",
    "electronic crowd",
    "music festival crowd",
    "music festival audience",
    "edm crowd",
    "rave crowd",
    "electronic music festival crowd",
    "techno festival audience",
]

PER_QUERY_N = 35          # 每个搜索词最多拉取多少个
MIN_DURATION = 30         # 最短时长（秒）
MIN_WIDTH, MIN_HEIGHT = 352, 240  # 最小分辨率
MAX_HEIGHT = 720          # 最高下载分辨率
DRYRUN = False            # True 时只预览搜索结果，不下载
# --------------------------------


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 数据根目录：{BASE_DIR}")
    print(f"[INFO] 原始视频：  {RAW_DIR}")
    print(f"[INFO] 清洗视频：  {CLEAN_DIR}")
    print(f"[INFO] 数据集目标：{DATASET_DIR}")


def ytdlp_options(out_dir: Path) -> Dict[str, Any]:
    fmt = f"mp4[height<={MAX_HEIGHT}]/best[ext=mp4]/best"
    opts = {
        "format": fmt,
        "nocheckcertificate": True,
        "ignoreerrors": True,
        "noplaylist": True,
        "no_warnings": True,
        "quiet": True,
        "outtmpl": str(out_dir / "%(title).80s-%(id)s.%(ext)s"),
        "writethumbnail": True,
        "writeinfojson": True,
        # 网络稳健性
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 15,
    }
    # 只有在支持的情况下才加 match_filter（按时长预过滤）
    if HAS_MATCH_FILTER:
        opts["match_filter"] = _match_filter_func(f"duration > {MIN_DURATION}")
    return opts


def download_search(query: str, n: int, out_dir: Path) -> int:
    url = f"ytsearch{n}:{query}"
    opts = ytdlp_options(out_dir)
    if DRYRUN:
        opts_preview = dict(opts)
        opts_preview["skip_download"] = True
        with YoutubeDL(opts_preview) as ydl:
            info = ydl.extract_info(url, download=False)
            cnt = len(info.get("entries", []) or [])
            print(f"  [PREVIEW] {query} -> {cnt} 条候选")
            return 0
    else:
        with YoutubeDL(opts) as ydl:
            print(f"  [DL] {query} …（{'含时长预过滤' if HAS_MATCH_FILTER else '无预过滤，后续清洗'}）")
            info = ydl.extract_info(url, download=True)
            cnt = len([e for e in (info.get("entries") or []) if e])
            print(f"  [DL] {query} 完成，候选统计≈{cnt}（以实际文件为准）")
            return cnt


def crawl_all():
    ensure_dirs()
    total = 0
    for q in KEYWORDS:
        try:
            total += download_search(q, PER_QUERY_N, RAW_DIR)
        except Exception as e:
            print(f"[WARN] 下载关键词失败：{q} | {e}")
    print(f"[INFO] 抓取流程结束（候选统计值）：{total}")


def video_meta(path: Path) -> Optional[Dict[str, float]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    dur = frames/float(fps) if fps > 0 else 0.0
    return {"fps": fps, "w": w, "h": h, "duration": dur}


def clean_videos():
    """质量清洗：最短时长/最低分辨率；复制到 CLEAN_DIR"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    vids = sorted([Path(p) for p in glob.glob(str(RAW_DIR / "*.mp4"))])
    print(f"[INFO] 原始 mp4 数量：{len(vids)}")

    kept = 0
    for vp in vids:
        m = video_meta(vp)
        if not m:
            print(f"[SKIP] 无法读取：{vp.name}")
            continue
        if m["duration"] < MIN_DURATION:
            print(f"[SKIP] 过短({m['duration']:.1f}s)：{vp.name}")
            continue
        if m["w"] < MIN_WIDTH or m["h"] < MIN_HEIGHT:
            print(f"[SKIP] 分辨率过低({m['w']}x{m['h']})：{vp.name}")
            continue

        dst = CLEAN_DIR / vp.name
        if not dst.exists():
            shutil.copy2(vp, dst)
        kept += 1
    print(f"[DONE] 通过清洗：{kept} 个 → {CLEAN_DIR}")


def build_dataset():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    vids = sorted([Path(p) for p in glob.glob(str(CLEAN_DIR / "*.mp4"))])
    print(f"[INFO] 清洗后视频：{len(vids)}")

    copied = 0
    for vp in vids:
        dst = DATASET_DIR / vp.name
        if not dst.exists():
            shutil.copy2(vp, dst)
            copied += 1
    print(f"[DONE] 数据集整理完成：复制 {copied} 个到 {DATASET_DIR}")
    print("\n[DATASET READY]")
    print(str(DATASET_DIR))


def main():
    print("=== STEP 1/3: 抓取视频（YouTube 搜索）===")
    crawl_all()
    print("\n=== STEP 2/3: 质量清洗（时长/分辨率）===")
    clean_videos()
    print("\n=== STEP 3/3: 建立数据集目录 ===")
    build_dataset()
    print("\n✅ 全流程完成！")


if __name__ == "__main__":
    main()
