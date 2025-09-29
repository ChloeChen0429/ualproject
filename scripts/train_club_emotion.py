#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-arg autorun trainer for club emotion.
- 数据源：~/Documents/ualproject/data/clean_videos（递归扫描）
- 模型输出：~/Documents/ualproject/models
- 步骤：YOLOv8 Pose 抽帧→特征→无监督聚类→以簇均值(A,V)映射 Russell 标签
- 产物：
  1) feats_windows_clean.csv  —— 每个窗口一条特征
  2) club_emotion_pipeline.joblib —— {scaler, pca, kmeans, cluster2label, feature_names}
  3) cluster_centers.csv —— 每个簇的 A/V/节奏统计
"""

import os, sys, time, glob, math, json, csv
from collections import deque, defaultdict
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# ----------------- 固定路径（无需参数） -----------------
HOME = os.path.expanduser("~")
VIDEOS_DIR = os.path.join(HOME, "Documents/ualproject/data/clean_videos")
OUT_DIR    = os.path.join(HOME, "Documents/ualproject/models")
FEATS_CSV  = os.path.join(HOME, "Documents/ualproject/data/feats_windows_clean.csv")
MODEL_FILE = os.path.join(OUT_DIR, "club_emotion_pipeline.joblib")
CLUSTER_CSV= os.path.join(OUT_DIR, "cluster_centers.csv")

# ----------------- 可调节配置 -----------------
POSE_WEIGHTS = "yolov8n-pose.pt"     # 也可换 yolov8s-pose.pt
FRAME_RATE_SAMPLE = 2.0              # 抽帧频率（fps）
MIN_AREA = 6000
KPT_CONF = 0.30
DEDUP_DIST = 35.0
EMA = 0.5
DEADBAND = 0.03
K_HEAD2SHOULDER = 2.4
WINDOW_SEC = 2.0                      # 统计窗口（秒）
MAX_PER_VIDEO_WINDOWS = None          # 可设为整数做采样
N_CLUSTERS = 6                        # 聚类簇数（4~8 试试）
RANDOM_STATE = 2024

# ----------------- 索引 -----------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = range(17)

# ----------------- 工具函数 -----------------
def angle_deg(a, b, c):
    v1 = np.array(a) - np.array(b); v2 = np.array(c) - np.array(b)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 90.0
    cos_ = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_)))

def center_of(pts):
    pts = np.array(pts, dtype=float); return float(np.mean(pts[:,0])), float(np.mean(pts[:,1]))

def norm_to_neg1_pos1(x, lo, hi):
    lo, hi = float(lo), float(hi)
    if hi <= lo: return 0.0
    y = (np.clip(float(x), lo, hi) - lo) / (hi - lo + 1e-9)
    return float(np.clip(y*2.0 - 1.0, -1.0, 1.0))

def estimate_scale_from_back(kpts_xy, box_xyxy, last_scale=None, k_head2shoulder=2.4):
    try:
        if kpts_xy is not None:
            if np.isfinite(kpts_xy[L_SHO]).all() and np.isfinite(kpts_xy[R_SHO]).all():
                sw = np.linalg.norm(kpts_xy[L_SHO]-kpts_xy[R_SHO])
                if sw > 1: return float(sw)
            if np.isfinite(kpts_xy[L_EAR]).all() and np.isfinite(kpts_xy[R_EAR]).all():
                headw = np.linalg.norm(kpts_xy[L_EAR]-kpts_xy[R_EAR])
                if headw > 1: return float(k_head2shoulder * headw)
            if np.isfinite(kpts_xy[L_EYE]).all() and np.isfinite(kpts_xy[R_EYE]).all():
                headw = np.linalg.norm(kpts_xy[L_EYE]-kpts_xy[R_EYE])
                if headw > 1: return float(k_head2shoulder * headw)
        if box_xyxy is not None:
            x1,y1,x2,y2 = [float(v) for v in box_xyxy]
            bw = max(2.0, x2-x1)
            headw_est = 0.3 * bw
            return float(k_head2shoulder * headw_est)
        if last_scale is not None: return float(last_scale)
    except Exception:
        pass
    return float(last_scale) if last_scale else 120.0

def match_tracks(prev_centers, curr_centers, max_dist=120.0):
    if len(prev_centers)==0 or len(curr_centers)==0: return {}
    prev_ids = list(prev_centers.keys())
    P = np.array([prev_centers[i] for i in prev_ids], dtype=float)
    C = np.array(curr_centers, dtype=float)
    cost = np.linalg.norm(P[:,None,:] - C[None,:,:], axis=2)
    r_ind, c_ind = linear_sum_assignment(cost)
    mapping = {}
    for ri, ci in zip(r_ind, c_ind):
        if cost[ri, ci] <= max_dist: mapping[ci] = prev_ids[ri]
    return mapping

def bbox_from_kp(kp):
    xs = kp[:,0]; ys = kp[:,1]
    xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
    if xs.size == 0 or ys.size == 0: return None
    x1, x2 = float(np.min(xs)), float(np.max(xs))
    y1, y2 = float(np.min(ys)), float(np.max(ys))
    pad = 0.06 * max(2.0, x2 - x1, y2 - y1)
    return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)

def map_to_av_club(features, hand_speed, vy_std, zcr_per_s):
    # A: 手速 + ZCR（节奏）+ 竖直速度波动
    A_spd = norm_to_neg1_pos1(hand_speed, 0.2, 3.0)
    A_zcr = norm_to_neg1_pos1(zcr_per_s, 0.3, 2.5)
    A_vy  = norm_to_neg1_pos1(vy_std,     0.10, 1.30)
    A     = float(np.clip(0.55*A_spd + 0.35*A_zcr + 0.10*A_vy, -1, 1))
    # V: 张开度 + 身体直立 + 肘更直更正
    V_open = norm_to_neg1_pos1(features.get("arm_open",0.0), 0.5, 2.6)
    V_arms = norm_to_neg1_pos1(features.get("elbows",90.0),   60.0, 175.0)
    V_trnk = norm_to_neg1_pos1(features.get("trunk_upright",0.5), 0.2, 1.0)
    V      = float(np.clip(0.55*V_open + 0.25*V_trnk + 0.20*V_arms, -1, 1))
    return A, V

def preprocess_frame_club(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v2 = clahe.apply(v)
    hsv2 = cv2.merge([h, s, v2])
    frame2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    mean_v = float(np.mean(v))
    gamma = 1.0
    if mean_v < 70: gamma = 0.9
    elif mean_v > 180: gamma = 1.1
    if abs(gamma - 1.0) > 1e-3:
        inv = 1.0 / max(1e-6, gamma)
        table = np.array([((i/255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        frame2 = cv2.LUT(frame2, table)
    frame2 = cv2.bilateralFilter(frame2, 5, 20, 20)
    return frame2

# ----------------- 特征提取（逐视频 → 多个窗口） -----------------
def extract_windows_from_video(path, model):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[WARN] 无法打开：{path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (5 <= fps <= 120): fps = 30.0
    sample_every = max(1, int(round(fps / FRAME_RATE_SAMPLE)))
    maxlen = int(WINDOW_SEC * (fps/ sample_every))  # 以抽帧频率估算窗口长度

    # 追踪状态
    next_id = 0
    prev_centers = {}
    id_last_scale = {}
    id_ema = {}
    id_wrist_hist = {}
    id_vy_hist = {}

    windows = []  # 每条: dict(视频名, 起止帧, A*_聚合, V*_聚合, 节奏/速度统计 ...)
    frame_idx = 0

    # 为了窗口化：缓存每帧的群体 A/V
    group_A_buf, group_V_buf, spd_buf, zcr_buf, open_buf = [], [], [], [], []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        frame_p = preprocess_frame_club(frame)
        res = model(frame_p, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4), dtype=float)
        scores= res.boxes.conf.cpu().numpy()  if (res.boxes is not None and res.boxes.conf is not None) else None
        kps_xy = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else np.zeros((0,17,2), dtype=float)
        kps_cf = res.keypoints.conf.cpu().numpy() if (res.keypoints is not None and hasattr(res.keypoints,'conf') and res.keypoints.conf is not None) else None

        cand = []
        num_kp = int(kps_xy.shape[0]); num_box = int(boxes.shape[0])
        has_scores = (scores is not None and len(scores) == num_box)
        has_conf_kp = (kps_cf is not None and (kps_cf.shape[0] == num_kp))

        for i in range(num_kp):
            kp = kps_xy[i]
            if np.isnan(kp).any(): continue
            if i < num_box:
                x1,y1,x2,y2 = boxes[i]
            else:
                bb = bbox_from_kp(kp)
                if bb is None: continue
                x1,y1,x2,y2 = bb
            area = max(0,(x2-x1)) * max(0,(y2-y1))
            if area < MIN_AREA: continue
            avgc = float(np.nanmean(kps_cf[i])) if has_conf_kp else 1.0
            if not np.isfinite(avgc) or avgc < KPT_CONF: continue
            cx, cy = float(np.mean(kp[:,0])), float(np.mean(kp[:,1]))
            cand.append({"idx": i, "kp": kp, "box": (x1,y1,x2,y2), "center": (cx,cy), "avgc": avgc})

        # 简单去重
        cand.sort(key=lambda d: (d["avgc"]), reverse=True)
        kept = []
        for c in cand:
            dup = False
            for f in kept:
                dx = c["center"][0]-f["center"][0]
                dy = c["center"][1]-f["center"][1]
                if (dx*dx+dy*dy)**0.5 < DEDUP_DIST:
                    dup = True; break
            if not dup: kept.append(c)

        curr_centers = [k["center"] for k in kept]
        assign = match_tracks(prev_centers, curr_centers, max_dist=120.0)
        for ci in range(len(curr_centers)):
            if ci not in assign: assign[ci] = next_id; next_id += 1

        now_t = time.time()
        A_list, V_list, spd_list, zcr_list, open_list = [], [], [], [], []

        for ci, who in enumerate(kept):
            pid = assign.get(ci)
            kp = who["kp"]; box = who["box"]
            scale = estimate_scale_from_back(kp, box, id_last_scale.get(pid), K_HEAD2SHOULDER)
            id_last_scale[pid] = scale

            def safe_pt(idx):
                return kp[idx] if np.isfinite(kp[idx]).all() else None
            Lw, Rw = safe_pt(L_WRI), safe_pt(R_WRI)
            Le, Re = safe_pt(L_ELB), safe_pt(R_ELB)
            Lsho, Rsho = safe_pt(L_SHO), safe_pt(R_SHO)

            # 头中心
            head_pts = [p for p in [safe_pt(L_EAR), safe_pt(R_EAR), safe_pt(L_EYE), safe_pt(R_EYE)] if p is not None]
            if len(head_pts)>=2:
                hx, hy = center_of(np.stack(head_pts, axis=0))
            else:
                x1,y1,x2,y2 = box; hx, hy = (x1+x2)/2.0, y1 + 0.15*(y2-y1)

            # 开放度
            if Lw is not None and Rw is not None:
                arm_open = np.linalg.norm(Lw - Rw) / max(1.0, scale)
            else:
                wr = Lw if Lw is not None else Rw
                arm_open = (abs(float(wr[0]-hx))*2.0 / max(1.0, scale)) if wr is not None else 0.0

            # 肘角
            def elbow_angle(elb, wri, sho_opt):
                if elb is None or wri is None: return 90.0
                a = sho_opt if sho_opt is not None else np.array([hx,hy], dtype=float)
                return angle_deg(a, elb, wri)
            L_ang = elbow_angle(Le, Lw, Lsho)
            R_ang = elbow_angle(Re, Rw, Rsho)
            elbows = (L_ang + R_ang)/2.0

            # 直立度
            x1,y1,x2,y2 = box
            chest_x, chest_y = (x1+x2)/2.0, y1 + 0.35*(y2-y1)
            v = np.array([hx - chest_x, hy - chest_y], dtype=float)
            theta_dev = abs(90.0 - abs(math.degrees(math.atan2(v[1], v[0])) - 90.0))
            trunk_upright = 1.0 - np.clip(theta_dev/90.0, 0.0, 1.0)

            # 时序（抽帧后以“抽帧节奏”近似）
            dq = id_wrist_hist.setdefault(pid, deque(maxlen=maxlen))
            Lx,Ly = (float(Lw[0]), float(Lw[1])) if Lw is not None else (np.nan, np.nan)
            Rx,Ry = (float(Rw[0]), float(Rw[1])) if Rw is not None else (np.nan, np.nan)
            dq.append((now_t, Lx,Ly, Rx,Ry))

            hand_speed = 0.0; vy_std = 0.0; vyy_all=[]
            if len(dq) >= 2:
                vs=[]
                for (t0,Lx0,Ly0,Rx0,Ry0),(t1,Lx1,Ly1,Rx1,Ry1) in zip(list(dq)[:-1], list(dq)[1:]):
                    dt = max(1e-3, t1-t0); sp = 0.0; vy=[]
                    if np.isfinite(Lx0) and np.isfinite(Lx1):
                        sp += math.hypot(Lx1-Lx0, Ly1-Ly0) / (scale*dt); vy.append((Ly1-Ly0)/max(1.0,scale))
                    if np.isfinite(Rx0) and np.isfinite(Rx1):
                        sp += math.hypot(Rx1-Rx0, Ry1-Ry0) / (scale*dt); vy.append((Ry1-Ry0)/max(1.0,scale))
                    if sp>0: vs.append(sp/(2 if len(vy)==2 else 1))
                    if len(vy)>0: vyy_all.extend(vy)
                if len(vs)>0: hand_speed = float(np.mean(vs))
                if len(vyy_all)>1: vy_std = float(np.std(vyy_all))

            # 节奏 ZCR
            vy_hist = id_vy_hist.setdefault(pid, deque(maxlen=maxlen))
            if len(vyy_all) > 0:
                vy_hist.append(np.mean(vyy_all))
            zcr_per_s = 0.0
            if len(vy_hist) > 4:
                arr = np.array(vy_hist, dtype=float)
                thr = max(0.02, 0.1*np.nanstd(arr))
                sign = np.sign(np.where(np.abs(arr) >= thr, arr, 0.0))
                cnt = 0
                for a,b in zip(sign[:-1], sign[1:]):
                    if a*b < 0: cnt += 1
                zcr_per_s = float(cnt / max(1e-6, WINDOW_SEC))

            # A/V（club 配置）
            feats = {"arm_open": float(arm_open), "elbows": float(elbows), "trunk_upright": float(trunk_upright)}
            A, V = map_to_av_club(feats, hand_speed, vy_std, zcr_per_s)

            # EMA + deadband
            A0,V0 = id_ema.get(pid, (A,V))
            A_s = A if abs(A-A0)>=DEADBAND else A0
            V_s = V if abs(V-V0)>=DEADBAND else V0
            A_ema = EMA*A0 + (1-EMA)*A_s
            V_ema = EMA*V0 + (1-EMA)*V_s
            id_ema[pid] = (A_ema, V_ema)

            A_list.append(A_ema); V_list.append(V_ema)
            spd_list.append(hand_speed); zcr_list.append(zcr_per_s); open_list.append(feats["arm_open"])

        # 每帧聚合为群体特征（抗异常）
        if len(A_list)>0:
            group_A = float(np.percentile(A_list, 75))
            group_V = float(np.median(V_list))
            group_A_buf.append(group_A)
            group_V_buf.append(group_V)
            if spd_list: spd_buf.append(float(np.median(spd_list)))
            if zcr_list: zcr_buf.append(float(np.median(zcr_list)))
            if open_list: open_buf.append(float(np.median(open_list)))

            # 满一窗就落一条
            if len(group_A_buf) >= maxlen:
                w = {
                    "video": os.path.basename(path),
                    "A_mean": float(np.mean(group_A_buf)),
                    "A_med":  float(np.median(group_A_buf)),
                    "A_p75":  float(np.percentile(group_A_buf, 75)),
                    "V_mean": float(np.mean(group_V_buf)),
                    "V_med":  float(np.median(group_V_buf)),
                    "V_p50":  float(np.percentile(group_V_buf, 50)),
                    "spd_med": float(np.median(spd_buf)) if spd_buf else 0.0,
                    "zcr_med": float(np.median(zcr_buf)) if zcr_buf else 0.0,
                    "open_med": float(np.median(open_buf)) if open_buf else 0.0,
                }
                windows.append(w)
                group_A_buf.clear(); group_V_buf.clear(); spd_buf.clear(); zcr_buf.clear(); open_buf.clear()

                if MAX_PER_VIDEO_WINDOWS and len(windows) >= MAX_PER_VIDEO_WINDOWS:
                    break

        prev_centers = { assign[i]: kept[i]["center"] for i in range(len(kept)) }

    cap.release()
    return windows

# ----------------- Russell 映射（基于簇 A/V 均值） -----------------
RUSSELL_SECTORS = [
    ("excited",   22.5),
    ("happy",     67.5),
    ("calm",     112.5),
    ("relaxed",  157.5),
    ("bored",    202.5),
    ("sad",      247.5),
    ("angry",    292.5),
    ("tense",    337.5),
]
def av_to_sector(A, V):
    r = math.hypot(V, A)
    if r < 0.15:
        return "neutral"
    theta = math.degrees(math.atan2(A, V))
    if theta < 0: theta += 360.0
    sectors = []
    last_deg = 0.0
    for name, boundary in RUSSELL_SECTORS:
        center = (last_deg + boundary) / 2.0
        sectors.append((name, center % 360.0))
        last_deg = boundary
    def ang_diff(a, b):
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)
    best = min(sectors, key=lambda x: ang_diff(theta, x[1]))
    return best[0]

# ----------------- 主流程（零参数 autorun） -----------------
def main():
    print("=== CLUB Emotion Trainer (autorun) ===")
    print(f"[INFO] videos_dir = {VIDEOS_DIR}")
    print(f"[INFO] out_dir    = {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FEATS_CSV), exist_ok=True)

    # 收集视频
    exts = (".mp4",".mov",".m4v",".mkv",".avi")
    videos = []
    for root,_,files in os.walk(VIDEOS_DIR):
        for f in files:
            if f.lower().endswith(exts):
                videos.append(os.path.join(root,f))
    videos.sort()
    if not videos:
        print("[ERR] clean_videos 目录未找到可用视频。")
        return

    print(f"[INFO] 共 {len(videos)} 个视频，将开始抽帧+提特征 …")
    model = YOLO(POSE_WEIGHTS)

    all_rows = []
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {os.path.basename(vp)}")
        ws = extract_windows_from_video(vp, model)
        for w in ws:
            all_rows.append(w)

    if not all_rows:
        print("[ERR] 未得到任何窗口特征，请检查视频/阈值。")
        return

    # 保存特征 CSV
    feat_cols = ["video","A_mean","A_med","A_p75","V_mean","V_med","V_p50","spd_med","zcr_med","open_med"]
    with open(FEATS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=feat_cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k:r.get(k,"") for k in feat_cols})
    print(f"[DONE] 特征已写入：{FEATS_CSV}（{len(all_rows)} 条窗口）")

    # 组装训练矩阵（无监督聚类）
    X = np.array([[r["A_mean"], r["A_med"], r["A_p75"],
                   r["V_mean"], r["V_med"], r["V_p50"],
                   r["spd_med"], r["zcr_med"], r["open_med"]] for r in all_rows], dtype=float)
    feature_names = ["A_mean","A_med","A_p75","V_mean","V_med","V_p50","spd_med","zcr_med","open_med"]

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    pca = PCA(n_components=min(5, Xn.shape[1]))
    Xp = pca.fit_transform(Xn)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    labels = kmeans.fit_predict(Xp)

    # 用每个簇的 A/V 均值→映射成 Russell 标签
    cluster2label = {}
    cluster_rows = []
    for c in range(N_CLUSTERS):
        idx = np.where(labels==c)[0]
        if idx.size == 0:
            cluster2label[c] = "neutral"
            cluster_rows.append({"cluster":c,"n":0,"A_mean":0,"V_mean":0,"label":"neutral"})
            continue
        A_mean = float(np.mean([all_rows[j]["A_mean"] for j in idx]))
        V_mean = float(np.mean([all_rows[j]["V_mean"] for j in idx]))
        lab = av_to_sector(A_mean, V_mean)
        cluster2label[c] = lab
        cluster_rows.append({"cluster":c,"n":int(idx.size),"A_mean":A_mean,"V_mean":V_mean,"label":lab})

    # 保存簇中心可读表
    with open(CLUSTER_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["cluster","n","A_mean","V_mean","label"])
        w.writeheader()
        for r in sorted(cluster_rows, key=lambda x:-x["n"]):
            w.writerow(r)
    print(f"[DONE] 聚类概览已写入：{CLUSTER_CSV}")

    # 保存推理管线
    payload = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "cluster2label": cluster2label,
        "feature_names": feature_names,
        "meta": {
            "videos_dir": VIDEOS_DIR,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "frame_rate_sample": FRAME_RATE_SAMPLE,
            "window_sec": WINDOW_SEC,
            "pose_weights": POSE_WEIGHTS
        }
    }
    joblib.dump(payload, MODEL_FILE)
    print(f"[DONE] 模型管线已保存：{MODEL_FILE}")

    print("\n[READY] 训练完成 ✅")
    print("你可以在实时交互代码里：")
    print("1) 按相同方式计算窗口特征（A/V/spd/zcr/open）")
    print("2) 用 joblib.load(模型) 取出 {scaler,pca,kmeans,cluster2label}")
    print("3) X → scaler → pca → kmeans.predict → cluster → 映射 cluster2label 成情绪标签")

if __name__ == "__main__":
    main()
