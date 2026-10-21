# -*- coding: utf-8 -*-
"""YOLO26-Seg 视频分割 + 稳定编号跟踪 + 离线平滑

设计要点
--------
1. 检测：best.pt（yolo26x-seg 微调），imgsz 与训练一致（2560x1440 -> 1280）。
2. 跟踪：BoT-SORT / DeepOCSORT，逐帧 persist，得到稳定底层 track_id。
3. 编号：每个类别独立计数，从 1 开始；编号**永久占用**，对象消失后编号作废不复用。
   同一帧内出现多个新对象时，按「右上角 -> 左下角」的空间顺序依次分配。
4. 去重：同一帧内同类别、位置高度重合的实例只保留一个（避免同像素多目标）。
5. 平滑：离线两遍。先收集整条轨迹，再对 (cx, cy, w, h) 与轮廓做时间域平滑 +
   短间隙插值，消除抖动与闪烁。不做实时，换取质量。

用法
----
    python tools/track_segment.py --source videos/2.mp4 --limit 400 --out out.mp4
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

CLS_NAMES = {0: "vehicle", 1: "rider", 2: "pedestrian"}
# BGR 配色：深蓝 / 深红 / 青绿
CLS_COLORS = {0: (121, 78, 31), 1: (0, 0, 192), 2: (143, 157, 42)}

CONTOUR_POINTS = 64  # 轮廓重采样点数（按弧长均匀）


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def order_key(cx, cy, mode):
    """编号分配的空间顺序键。越小越先编号。

    diag: 从右上角向左下角推进（对角扫描）—— 右上 x 大 y 小，x-y 最大
    row : 从上到下逐行，行内从右到左
    """
    if mode == "row":
        return (cy, -cx)
    return (cy - cx, cy)


def contour_to_poly(mask_bool, eps_ratio=0.004):
    """二值掩码 -> 简化后的外轮廓点 (N,2) float32；失败返回 None。"""
    m = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 12:
        return None
    peri = cv2.arcLength(c, True)
    ap = cv2.approxPolyDP(c, eps_ratio * peri, True)
    return ap.reshape(-1, 2).astype(np.float32)


def poly_resample(poly, n=CONTOUR_POINTS):
    """把多边形按弧长均匀重采样成 n 个点（固定点数，便于时间域平滑）。"""
    p = np.asarray(poly, np.float32)
    if len(p) < 3:
        return None
    d = np.sqrt(((np.roll(p, -1, axis=0) - p) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-6:
        return None
    t = np.linspace(0, total, n, endpoint=False)
    x = np.interp(t, s[:-1], p[:, 0])
    y = np.interp(t, s[:-1], p[:, 1])
    return np.stack([x, y], 1).astype(np.float32)


def mask_to_poly(mask, box, eps_ratio=0.004):
    """只在 bbox 邻域内提取轮廓（比全图 findContours 快一个量级）。"""
    H, W = mask.shape[:2]
    x1 = int(max(0, np.floor(box[0]) - 3))
    y1 = int(max(0, np.floor(box[1]) - 3))
    x2 = int(min(W, np.ceil(box[2]) + 3))
    y2 = int(min(H, np.ceil(box[3]) + 3))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    poly = contour_to_poly(mask[y1:y2, x1:x2] > 0.5, eps_ratio)
    if poly is None:
        return None
    return poly_resample(poly + np.array([x1, y1], np.float32))


# ----------------------------------------------------------------------------
# Pass 1：逐帧检测 + 跟踪，收集轨迹
# ----------------------------------------------------------------------------
def pass1_collect(model, src, imgsz, conf, tracker, limit, dedup_ratio, order,
                  half=False, dedup_cross=0.8, verbose_every=200, cls_tol=30,
                  dedup_dup=0.9, conf_init=0.8, device="0"):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    todo = min(total, limit) if limit else total

    tracks = {}          # tid -> dict(cls, num, fi[], box[], conf[], poly[])
    used_numbers = {}    # cls -> set(已用编号)
    next_number = {}     # cls -> 下一个待分配编号
    stats = {"frames": 0, "raw": 0, "dedup": 0, "new_track": 0}
    rows_seen = []       # 用于自动检测上下黑边（全片分散采样）
    sample_step = max(1, todo // 30)

    t0 = time.time()
    for fi in range(todo):
        ok, frame = cap.read()
        if not ok:
            break
        if fi % sample_step == 0 and len(rows_seen) < 40:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows_seen.append(g.max(axis=1) >= 12)
        r = model.track(frame, persist=True, tracker=tracker, imgsz=imgsz,
                        conf=conf, retina_masks=True, device=device,
                        quantize=(16 if half else 32),
                        verbose=False)[0]
        stats["frames"] += 1
        if stats["frames"] == 1:
            # 记录首帧张量实际落在哪个设备上 —— 这是"到底跑没跑 GPU"的硬证据
            try:
                dev = r.boxes.data.device if r.boxes is not None else None
                stats["device"] = str(dev) if dev is not None else "n/a"
            except Exception:
                stats["device"] = "n/a"
        b = r.boxes
        if b is None or len(b) == 0 or b.id is None:
            continue

        ids = b.id.cpu().numpy().astype(int)
        clss = b.cls.cpu().numpy().astype(int)
        cfs = b.conf.cpu().numpy().astype(np.float32)
        xyxy = b.xyxy.cpu().numpy().astype(np.float32)
        masks = r.masks.data.cpu().numpy() if r.masks is not None else None
        stats["raw"] += len(ids)

        # ---- 重复实例去重（**身份感知 + 保连续性**）
        #
        # 原实现只用「几何重叠 + 置信度」判定重复，有两个结构性缺陷：
        #
        #   缺陷1 不认跟踪身份：两个**不同的**已建轨迹（不同 tid）只要框重叠超阈值
        #     就会互相去重。本场景目标密集、框大量交叠（相邻 rider 的框常常重叠
        #     过半），于是互删频繁发生。
        #
        #   缺陷2 按置信度决定删谁：可能删掉「某条已建轨迹正在续上的那一帧」，
        #     转而保留一个刚冒出来的新检测 —— 已建轨迹被硬生生切断。
        #
        #   最严重的是它作用在**轨迹出生帧**上：`new_idx` 是在去重后的 `sel` 上
        #   算的，所以某目标的检测若在出生帧被去重掉，它的 tid 根本不会进入
        #   `tracks` —— 表现为该对象**整段完全消失**（而不是中间有缺口）。
        #
        # 修正原则：
        #   (a) 两个都在 `tracks` 里、且 tid 不同 -> 是两个已确认的不同对象，不去重；
        #   (b) 一对里只有一个已建轨迹且 tid 不同 -> 去重，但**必须删掉没建轨的那个**，
        #       保住已建轨迹的连续性（不再看置信度高低）；
        #   (c) 其余情况（同 tid 重复、双方都是新检测）才按置信度删低分那个。
        keep = np.ones(len(ids), bool)
        for i in range(len(ids)):
            if not keep[i]:
                continue
            ti = int(ids[i])
            for j in range(i + 1, len(ids)):
                if not keep[j]:
                    continue
                ci, cj = int(clss[i]), int(clss[j])
                if ci == cj:
                    thr = dedup_ratio
                elif {ci, cj} <= {1, 2}:          # rider / pedestrian 之间的混淆
                    thr = dedup_cross
                else:
                    continue
                tj = int(ids[j])
                est_i = ti in tracks
                est_j = tj in tracks
                a, c = xyxy[i], xyxy[j]
                iw = max(0.0, min(a[2], c[2]) - max(a[0], c[0]))
                ih = max(0.0, min(a[3], c[3]) - max(a[1], c[1]))
                inter = iw * ih
                ar = min((a[2] - a[0]) * (a[3] - a[1]), (c[2] - c[0]) * (c[3] - c[1]))
                ratio = (inter / ar) if ar > 0 else 0.0
                if ratio <= thr:
                    continue
                # (a') 两个已确认轨迹：只有**几乎完全重合**才算同一物体的重复检测。
                # 直接用 dedup_ratio(0.5) 会在密集场景里让相邻的不同 rider 互删
                # （本场景框大量交叠，实测误删频发）；但完全不去重又会放过跟踪器
                # 给同一物体发的两个 id（实测这类重复的重叠率 0.977~0.992）。
                # 所以这里用更高的 dedup_dup(默认 0.9) 把两种情况分开。
                if ti != tj and est_i and est_j and ratio <= dedup_dup:
                    continue
                if ti != tj and est_i != est_j:
                    drop = j if est_i else i   # (b) 删没建轨的那个
                    # 记一笔：本次去重**保住**了一条已建轨迹的连续帧
                    stats["dedup_keep_est"] = stats.get("dedup_keep_est", 0) + 1
                else:
                    drop = j if cfs[i] >= cfs[j] else i
                keep[drop] = False
                stats["dedup"] += 1
                if ci != cj:
                    stats["dedup_cross"] = stats.get("dedup_cross", 0) + 1
                if drop == i:
                    break

        sel = np.where(keep)[0]
        if len(sel) == 0:
            continue

        # ---- 新 track 按空间顺序分配编号（同帧内一起排）
        #
        # ★ 首现置信度门限（conf_init）：
        #   检测本身跑在较低的 conf（跟踪门限）上，好让已建轨迹在被遮挡/模糊时
        #   仍有检测可续；但**新对象的出生**要求更高分，避免低分误检凭空建立
        #   一条新轨迹（这类误检会长期占编号、污染统计）。
        #   已建轨迹的续帧不受此限，所以不会打断已有身份。
        new_idx = [k for k in sel
                   if int(ids[k]) not in tracks and float(cfs[k]) >= conf_init]
        if new_idx:
            def key(k):
                a = xyxy[k]
                return order_key((a[0] + a[2]) / 2, (a[1] + a[3]) / 2, order)
            for k in sorted(new_idx, key=key):
                tid = int(ids[k])
                cl = int(clss[k])
                if tid in tracks:      # 极端情况下同帧重复 id
                    continue
                n = next_number.get(cl, 1)
                while n in used_numbers.get(cl, set()):
                    n += 1
                used_numbers.setdefault(cl, set()).add(n)
                next_number[cl] = n + 1
                tracks[tid] = {"cls": cl, "num": n, "fi": [], "box": [],
                               "conf": [], "poly": [], "mism": 0}
                stats["new_track"] += 1

        # ---- 收集本帧结果
        #
        # 类别跳变：原实现是「本帧类别 != 轨迹出生时的类别 -> 整帧丢弃」。
        # 问题在于这些目标（尤其原地打转的 rider）在分类器眼里会在 rider /
        # pedestrian 之间反复横跳，于是被一帧一帧地丢，轨迹被切成碎片、
        # 甚至完全空掉。而跟踪器既然给了同一个 tid，就说明它认为这是同一个
        # 对象 —— 掩膜几何仍然有效，类别只是标签。
        # 修正：容忍**连续**跳变（cls_tol 帧），保留几何并沿用轨迹自身类别；
        # 连续跳变超过 cls_tol 才停收（那时更可能是发生了 ID 切换）。
        for k in sel:
            tid = int(ids[k])
            if tid not in tracks:
                continue
            tr = tracks[tid]
            if tr["cls"] != int(clss[k]):
                tr["mism"] += 1
                if tr["mism"] > cls_tol:
                    continue                      # 长时间不一致 -> 疑为 ID 切换
                stats["cls_tol"] = stats.get("cls_tol", 0) + 1
            else:
                tr["mism"] = 0
            tr["fi"].append(fi)
            tr["conf"].append(float(cfs[k]))
            tr["box"].append(xyxy[k].copy())
            if masks is not None and k < len(masks):
                tr["poly"].append(mask_to_poly(masks[k], xyxy[k]))
            else:
                tr["poly"].append(None)

        if verbose_every and (fi + 1) % verbose_every == 0:
            el = time.time() - t0
            print(f"    {fi+1}/{todo} | {el:.0f}s | {(fi+1)/el:.1f} fps | "
                  f"tracks {len(tracks)} | eta {(todo-fi-1)/max((fi+1)/el,1e-9):.0f}s",
                  flush=True)

    cap.release()
    # 自动检测画面有效行范围（上下黑边）：全片分散采样 + 多数投票，
    # 避免视频开头的黑帧/转场把整幅画面判成黑边。
    if rows_seen:
        M = np.stack(rows_seen)                       # (n, H) bool
        keep = M.sum(axis=1) > 50                     # 丢掉全黑帧
        if keep.sum() >= 3:
            M = M[keep]
        frac = M.mean(axis=0)                         # 该行"有内容"的帧占比
        idx = np.where(frac > 0.5)[0]
        if len(idx) > 100:
            eff_top, eff_bot = int(idx[0]), int(idx[-1])
        else:
            eff_top, eff_bot = 0, H - 1
    else:
        eff_top, eff_bot = 0, H - 1
    stats["elapsed"] = time.time() - t0
    stats["eff_rows"] = [eff_top, eff_bot]
    return tracks, stats, (W, H, fps, total, eff_top, eff_bot)


# ----------------------------------------------------------------------------
# 轨迹缝合：把「同一对象因短暂丢失而被拆成两段」的轨迹接回同一条
# ----------------------------------------------------------------------------
def _vel(box, fi, head=False, k=12):
    """用首/尾 k 帧线性拟合估计速度 (px/帧)，返回 (vx, vy)。"""
    n = len(fi)
    if n < 2:
        return np.zeros(2)
    if head:
        idx = np.arange(min(k, n))
    else:
        idx = np.arange(max(0, n - k), n)
    cx = (box[idx, 0] + box[idx, 2]) / 2
    cy = (box[idx, 1] + box[idx, 3]) / 2
    t = np.asarray(fi)[idx].astype(np.float64)
    t = t - t.mean()
    if len(t) < 2 or np.ptp(t) < 1e-6:
        return np.zeros(2)
    vx = float(np.polyfit(t, cx, 1)[0])
    vy = float(np.polyfit(t, cy, 1)[0])
    return np.array([vx, vy])


def _ends(tr):
    box = np.asarray(tr["box"], np.float32)
    fi = np.asarray(tr["fi"], np.int32)
    s = np.array([(box[0, 0] + box[0, 2]) / 2, (box[0, 1] + box[0, 3]) / 2])
    e = np.array([(box[-1, 0] + box[-1, 2]) / 2, (box[-1, 1] + box[-1, 3]) / 2])
    return fi, s, e, _vel(box, fi, head=True), _vel(box, fi, head=False)


def _moving_out(p, v, border, vmin=0.3):
    """点位于边界带且速度指向画面外 —— 即"正在走出画面"。

    只看位置不看方向会把「贴着画面边缘向内走」的对象也误判成出画，
    所以必须结合速度方向。
    """
    if border is None:
        return False
    x0, y0, x1, y1 = border
    if p[0] >= x1 and v[0] > vmin:
        return True                      # 右边 + 继续向右
    if p[0] <= x0 and v[0] < -vmin:
        return True                      # 左边 + 继续向左
    if p[1] <= y0 and v[1] < -vmin:
        return True                      # 上边 + 继续向上
    if p[1] >= y1 and v[1] > vmin:
        return True                      # 下边 + 继续向下
    return False


def _in_border_band(p, border):
    """点是否落在画面边界带内 —— **只看位置，不看速度方向**。

    取代缝合里原来对 `_moving_out` 的调用：`_moving_out` 还要判断"速度是否指向
    画面外"，那属于方向判断，已按要求去掉。这里退化成纯位置判据 ——
    轨迹端点只要贴到边界带上就不参与缝合，因为它更可能是正在离场/刚进场的对象，
    与其缝合容易把两个不同对象接成一条。

    说明：`_moving_out` 本身保留，`audit_tracks.py` 的断裂判定仍在用它。
    """
    if border is None:
        return False
    x0, y0, x1, y1 = border
    return bool(p[0] >= x1 or p[0] <= x0 or p[1] <= y0 or p[1] >= y1)


def stitch_tracks(tracks, max_gap=30, max_res=80.0, max_angle=60.0,
                  max_speed=120.0, border=None, verbose=True):
    """迭代合并同类别中「尾部 -> 头部」时间接近且位置吻合的轨迹。

    判定：**位置外推残差 <= max_res** 且 速度模长合理。

    关于方向判断（已移除）
    ----------------------
    原逻辑还要求「前段末速度」与「后段初速度」的夹角 <= max_angle。这是运动
    方向判断，已按要求去掉：只保留"用前段末速度把位置外推到 gap 帧后，看是否
    落在后段起点附近"这一条**位置**判据。`max_angle` 参数保留仅为兼容旧调用方
    （`mask_ab.py` / `verify_mask_fix.py` 仍在传）。

    边界处理也一并改成纯位置判据 `_in_border_band`：端点只要贴在画面边界带上
    就不参与缝合（可能是在离场/刚进场的对象），不再看速度是否指向画面外。
    合并后保留前段编号，后段编号作废（不再分配给其他对象）。
    """
    merged_log = []
    changed = True
    while changed:
        changed = False
        info = {t: _ends(tr) for t, tr in tracks.items()}
        best = None
        for a, (fia, sa, ea, va, _) in info.items():
            if _in_border_band(ea, border):
                continue                 # 前段端点贴边界 -> 可能在离场，不缝
            for b, (fib, sb, eb, _, vb) in info.items():
                if a == b or tracks[a]["cls"] != tracks[b]["cls"]:
                    continue
                if _in_border_band(sb, border):
                    continue             # 后段起点就贴边界 -> 可能刚进场，不缝
                gap = int(fib[0]) - int(fia[-1])
                if gap <= 0 or gap > max_gap:
                    continue
                if np.linalg.norm(va) > max_speed or np.linalg.norm(vb) > max_speed:
                    continue
                # 只用位置外推 + 残差门限；**不做速度方向夹角判断**
                pred = ea + va * gap
                res = float(np.linalg.norm(pred - sb))
                if res > max_res:
                    continue
                score = res + gap * 1.0
                if best is None or score < best[0]:
                    best = (score, a, b, gap, res)
        if best is not None:
            _, a, b, gap, res = best
            ta, tb = tracks[a], tracks[b]
            ta["fi"].extend(tb["fi"])
            ta["box"].extend(tb["box"])
            ta["conf"].extend(tb["conf"])
            ta["poly"].extend(tb["poly"])
            merged_log.append({"from": tb["num"], "into": ta["num"],
                               "cls": CLS_NAMES[ta["cls"]], "gap": gap,
                               "residual": round(res, 1)})
            tracks.pop(b)
            changed = True
    if verbose and merged_log:
        print(f"[stitch] 缝合 {len(merged_log)} 段轨迹（编号保持前段）", flush=True)
        for m in merged_log[:12]:
            print(f"         {m['cls']:11s} #{m['from']:<3d} -> #{m['into']:<3d} "
                  f"gap {m['gap']:3d}f  残差 {m['residual']:6.1f}px", flush=True)
    return tracks, merged_log


# ----------------------------------------------------------------------------
# 掩膜轮廓的时序处理
#
# 轮廓按弧长重采样成固定 64 点后，每帧的「第 i 个点」并不对应同一个物理
# 位置（起始点会漂移、形状会变形）。若直接对第 i 个点做时间平滑，等于把
# 不对应的点平均到一起 —— 表现为多边形突然塌缩成很小的几何、或鼓出框外。
#
# 因此顺序必须是：
#   框归一化 -> 循环位移对齐 -> 时间平滑 -> 形状变化限速 -> 收进框内 -> 反投影
# ----------------------------------------------------------------------------
def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _box_norm_poly(p, box):
    """轮廓 -> 框归一化偏移（框中心为原点、框宽高为单位）。"""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    bw = max(float(box[2] - box[0]), 1.0)
    bh = max(float(box[3] - box[1]), 1.0)
    return np.stack([(p[:, 0] - cx) / bw, (p[:, 1] - cy) / bh], 1).astype(np.float64)


def _roll_med(y, win):
    """滚动中位数（边缘复制填充）。"""
    y = np.asarray(y, np.float64)
    n = len(y)
    if win < 3 or n < 3:
        return y.copy()
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 3:
        return y.copy()
    half = win // 2
    yp = np.pad(y, half, mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(yp, win)
    return np.median(sw, axis=-1)


def _align_stack(qs):
    """把一串框归一化轮廓按循环位移对齐到同一参数化起点。

    先链式对齐（每帧对上一帧），再以中位形状为模板迭代细化。
    返回 (M, N, 2)。
    """
    n = len(qs[0])
    ar = np.arange(n)
    idx = (ar[:, None] + ar[None, :]) % n

    def unit(q):
        u = q - q.mean(0)
        s = float(np.linalg.norm(u, axis=1).mean())
        return u / (s if s > 1e-9 else 1.0)

    def shift(ref, uq):
        """返回 k，使 np.roll(uq, k) 最接近 ref。"""
        M = ((uq[:, None, :] - ref[None, :, :]) ** 2).sum(-1)     # (n,n)
        return int(np.argmin(M[ar[:, None], idx].sum(0)))

    out = [qs[0]]
    for i in range(1, len(qs)):
        out.append(np.roll(qs[i], shift(unit(out[-1]), unit(qs[i])), axis=0))
    for _ in range(2):
        ref = unit(np.median(np.stack(out), 0))
        out = [np.roll(p, shift(ref, unit(p)), axis=0) for p in out]
    return np.stack(out, 0)


def _limit_shape(qx, qy, max_dev):
    """限制相邻帧之间轮廓点的平均位移（框归一化单位），抑制形状突变。

    正向、反向各限幅一遍再取平均，避免单向拖尾。
    """
    if max_dev <= 0:
        return qx, qy
    T = len(qx)

    def sweep(sx, sy):
        ox, oy = sx.copy(), sy.copy()
        for i in range(1, T):
            d = float(np.sqrt((sx[i] - ox[i - 1]) ** 2
                              + (sy[i] - oy[i - 1]) ** 2).mean())
            if d > max_dev:
                s = max_dev / d
                ox[i] = ox[i - 1] + (sx[i] - ox[i - 1]) * s
                oy[i] = oy[i - 1] + (sy[i] - oy[i - 1]) * s
        return ox, oy

    fx, fy = sweep(qx, qy)
    bx, by = sweep(qx[::-1].copy(), qy[::-1].copy())
    return (fx + bx[::-1]) / 2.0, (fy + by[::-1]) / 2.0


def _clamp_q(q, tol):
    """把轮廓顶点沿「轮廓质心 -> 顶点」的射线收进 [-0.5-tol, 0.5+tol]²。

    若收边会把轮廓压塌（面积掉到 40% 以下），说明质心已在框外，放弃收边。
    """
    c = q.mean(0)
    d = q - c
    t_lim = np.ones(len(q))
    for k in (0, 1):
        lo, hi = -0.5 - tol, 0.5 + tol
        dk = d[:, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hi = np.where(dk > 1e-9, (hi - c[k]) / dk, np.inf)
            t_lo = np.where(dk < -1e-9, (lo - c[k]) / dk, np.inf)
        t_k = np.minimum(np.where(t_hi > 0, t_hi, np.inf),
                         np.where(t_lo > 0, t_lo, np.inf))
        t_lim = np.minimum(t_lim, np.maximum(t_k, 0.0))
    out = c + d * np.clip(t_lim, 0.0, 1.0)[:, None]
    a0, a1 = _poly_area(q), _poly_area(out)
    if a0 > 1e-9 and a1 < 0.4 * a0:
        return q
    return out


# ----------------------------------------------------------------------------
# 掩膜伪影清理：① 近邻小掩膜突变为一个「大掩膜」  ② 两掩膜突发高重叠
#
# 两者的处理思路一致：**只丢这一帧的观测，不改编号、不动前后帧**。
# 丢掉之后，`smooth_tracks` 会在 gap（默认 <=max_gap）内做线性插值，
# 于是这一帧被「重算」成由前后帧外推出来的形状 —— 也就是把那一帧
# 重新识别了一遍，而不是相信那个坏检测。
# ----------------------------------------------------------------------------
def _poly_overlap(p, q):
    """两个多边形的重叠像素数与各自面积（在联合包围盒内栅格化）。"""
    if p is None or q is None or len(p) < 3 or len(q) < 3:
        return 0.0, 0.0, 0.0
    x1 = int(np.floor(min(p[:, 0].min(), q[:, 0].min())))
    y1 = int(np.floor(min(p[:, 1].min(), q[:, 1].min())))
    x2 = int(np.ceil(max(p[:, 0].max(), q[:, 0].max()))) + 1
    y2 = int(np.ceil(max(p[:, 1].max(), q[:, 1].max()))) + 1
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return 0.0, 0.0, 0.0
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    off = np.array([x1, y1], np.float32)
    cv2.fillPoly(m1, [np.round(p - off).astype(np.int32)], 1)
    cv2.fillPoly(m2, [np.round(q - off).astype(np.int32)], 1)
    return (float(np.count_nonzero(m1 & m2)),
            float(np.count_nonzero(m1)), float(np.count_nonzero(m2)))


def _box_cover(inner, outer):
    """inner 框被 outer 框覆盖的面积比（以 inner 自身面积为分母）。"""
    iw = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    ih = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    ia = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (iw * ih) / ia if ia > 1e-6 else 0.0


def _drop_positions(tr, positions):
    """按位置索引删除轨迹的若干帧（fi/box/conf/poly 四个并列列表同步删）。"""
    for i in sorted(set(int(p) for p in positions), reverse=True):
        for k in ("fi", "box", "conf", "poly"):
            if i < len(tr[k]):
                del tr[k][i]


def fix_merged_masks(tracks, area_ratio=1.45, cover=0.5, look=3,
                     min_neighbors=2, warmup=5, max_len=12, verbose=True):
    """① 近邻若干小掩膜突变为一个大掩膜时，清掉大掩膜、还原原来的小掩膜。

    判定「吞并」：本帧有 >= `min_neighbors` 条别的轨迹缺席，且它们在 `look`
    帧内刚消失、末框 >= `cover` 的比例落在本框内（或同帧共存但被本掩膜包住），
    同时本掩膜面积明显大于被吞的那些。

    ★ 处理方式取决于这个大掩膜**落在谁身上**（实测踩出来的关键区别）：

      A. 落给**已有轨迹** —— 该轨迹本帧面积相对自身历史骤增。
         只丢这一帧观测；前后帧都在，插值会把这一帧重建回原本的小形状。
         这才是真正"还原成原来的几个小的"。

      B. 落给**一条新轨迹**（出生帧就是那个大掩膜）。
         此时**不能只丢帧** —— 新轨迹没有历史可插值，丢了等于对象凭空消失
         （实测：f1259 一个真实行人就这样整帧没了掩膜）。
         正确做法：若这条轨迹是**短暂的**（<= `max_len` 帧，粘连本就是瞬时的），
         整条删掉；被它吞掉的近邻有前后历史，插值会把它们补回来。

    长轨迹且面积未骤增 -> 可能真是个大目标，**不动**（宁可漏修，不误删）。
    """
    if not tracks:
        return 0
    # 逐帧掩膜面积（多边形面积；无多边形时用框面积 × 0.7 近似）
    area = {}
    idx_of = {}
    for tid, tr in tracks.items():
        a = []
        for p, b in zip(tr["poly"], tr["box"]):
            if p is not None and len(p) >= 3:
                a.append(_poly_area(p))
            else:
                a.append(float((b[2] - b[0]) * (b[3] - b[1])) * 0.7)
        area[tid] = np.asarray(a, np.float64)
        idx_of[tid] = {int(f): i for i, f in enumerate(tr["fi"])}

    per = {}
    for tid, tr in tracks.items():
        for i, f in enumerate(tr["fi"]):
            per.setdefault(int(f), []).append((tid, i))

    def vanished_hosts(tid, i, f):
        """本帧吞掉的「刚消失」近邻 -> [(oid, 其末帧面积)]。"""
        box_i = tracks[tid]["box"][i]
        out = []
        for oid, otr in tracks.items():
            if oid == tid or f in idx_of[oid]:
                continue
            ofi = np.asarray(otr["fi"], np.int32)
            k = int(np.searchsorted(ofi, f)) - 1
            if k < 0 or f - int(ofi[k]) > look:
                continue
            if _box_cover(otr["box"][k], box_i) >= cover:
                out.append((oid, float(area[oid][k])))
        return out

    def coexisting_inside(tid, i, f):
        """本帧同类别、且 >=70% 像素落在本掩膜内的更小掩膜 -> [(oid, 面积)]。"""
        tr = tracks[tid]
        pa = tr["poly"][i]
        if pa is None or len(pa) < 3:
            return []
        out = []
        for oid, j in per.get(f, ()):
            if oid == tid or tracks[oid]["cls"] != tr["cls"]:
                continue
            if area[oid][j] * 1.3 >= area[tid][i]:
                continue
            pb = tracks[oid]["poly"][j]
            if pb is None or len(pb) < 3:
                continue
            it, _, a2 = _poly_overlap(pa, pb)
            if a2 > 1e-6 and it / a2 >= 0.7:
                out.append((oid, float(area[oid][j])))
        return out

    drop_frames, kill, events = {}, set(), []
    for tid, tr in tracks.items():
        fi = tr["fi"]
        ar = area[tid]
        transient = len(fi) <= max_len
        for i in range(len(fi)):
            f = int(fi[i])
            born = i < warmup
            inflated = False
            if not born:
                med = float(np.median(ar[max(0, i - 15):i]))
                inflated = med > 1.0 and ar[i] > area_ratio * med
            if not (born or inflated):
                continue
            hosts = vanished_hosts(tid, i, f)
            if len(hosts) < min_neighbors:
                hosts = hosts + coexisting_inside(tid, i, f)
            if len(hosts) < min_neighbors:
                continue
            if ar[i] <= 1.2 * float(np.median([h[1] for h in hosts])):
                continue                       # 没明显比被吞的那些大 -> 不是粘连
            if transient:
                # 新出生/短暂的大掩膜：**整条删掉**。只丢帧没用 —— 它没有历史
                # 可插值，丢了等于对象凭空消失；删整条后，被它吞掉的近邻本来
                # 有前后历史，插值会把它们还原成原来的几个小掩膜。
                kill.add(tid)
                events.append({"f": f, "cls": CLS_NAMES[tr["cls"]], "num": tr["num"],
                               "area": float(ar[i]), "n": len(hosts),
                               "mode": f"删整条({len(fi)}帧)"})
                break
            if inflated:
                # 已有轨迹被撑大：只丢这一帧，插值会把它重建回原本的小形状
                drop_frames.setdefault(tid, []).append(i)
                events.append({"f": f, "cls": CLS_NAMES[tr["cls"]], "num": tr["num"],
                               "area": float(ar[i]), "n": len(hosts),
                               "mode": "丢该帧"})
            # 长轨迹且未骤增 -> 可能真是个大目标，不动（宁可漏修，不误删）

    for tid in kill:
        tracks.pop(tid, None)
    for tid, pos in drop_frames.items():
        tr = tracks.get(tid)
        if tr is None:
            continue
        _drop_positions(tr, pos)
    for tid in [t for t, v in tracks.items() if not v["fi"]]:
        tracks.pop(tid)

    if verbose and events:
        nk = sum(1 for e in events if e["mode"].startswith("删"))
        print(f"[fix-merge] 掩膜粘连修正 {len(events)} 处"
              f"（整条删除 {nk} / 只丢单帧 {len(events) - nk}）"
              f"—— 原小掩膜由插值还原", flush=True)
        for e in events[:8]:
            print(f"         {e['cls']:11s} #{e['num']:<3d} f{e['f']:<5d} "
                  f"面积 {e['area']:7.0f} 吞并 {e['n']} 个近邻  [{e['mode']}]",
                  flush=True)

    return len(events)

def fix_overlap_conflicts(tracks, jump=0.80, base_thr=0.25, window=15,
                          drop_conf=0.6, min_hist=4, center_ratio=0.35,
                          verbose=True):
    """② 两个掩膜在某帧**突发**高度重叠时，丢掉其中置信度较低的那个。

    ★★ 为什么门槛必须卡在「重复检测」级别，而不是任意"重叠较多"：
       实测（14.mp4）：两个**真实相邻**的行人，掩膜重叠可达 56%~70%，
       质心只差 20~30px（相当于掩膜尺度的 0.5 倍）。按 0.55 的门限去丢，
       会把真人整帧删掉（f2603 的白衣行人就这样没了掩膜）。
       所以这里要求 **同时** 满足：
         a. 重叠比（相对较小的那个掩膜）>= `jump`（默认 0.80，接近完全重合）；
         b. 质心距 / sqrt(小掩膜面积) <= `center_ratio`（默认 0.35，即几乎同一个位置）；
         c. 相对这一对**自己的历史中位**是突跳（r_f - base >= `base_thr`），
            排除"本来就一直紧贴"的固定搭档。
       三者同时满足才判为同一物体被重复检出。

    丢谁：
      1) 一方 conf < `drop_conf`、另一方不低于 -> 丢低分那个；
      2) 两边都 < `drop_conf` -> 丢 conf 更低的；
      3) 两边都不低 -> 丢掩膜面积更小的那个（"更小的那一个"）。
    """
    if not tracks or len(tracks) < 2:
        return 0
    per = {}
    for tid, tr in tracks.items():
        for i, f in enumerate(tr["fi"]):
            per.setdefault(int(f), []).append((tid, i))
    if not per:
        return 0

    idx_of = {tid: {int(f): i for i, f in enumerate(tr["fi"])}
              for tid, tr in tracks.items()}

    # ---- 候选对：逐帧框重叠粗筛（numpy 矩阵化，避免两两 Python 循环）----
    cand = set()
    for f, lst in per.items():
        if len(lst) < 2:
            continue
        bx = np.asarray([tracks[t]["box"][i] for t, i in lst], np.float32)
        areas = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
        iw = np.minimum(bx[:, None, 2], bx[None, :, 2]) - \
            np.maximum(bx[:, None, 0], bx[None, :, 0])
        ih = np.minimum(bx[:, None, 3], bx[None, :, 3]) - \
            np.maximum(bx[:, None, 1], bx[None, :, 1])
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        rel = inter / np.minimum(areas[:, None], areas[None, :]).clip(1e-6)
        ii, jj = np.nonzero(np.triu(rel >= 0.3, 1))
        for x, y in zip(ii, jj):
            a, b = lst[int(x)][0], lst[int(y)][0]
            cand.add((a, b) if a < b else (b, a))
    if not cand:
        return 0

    drops, events = {}, []
    for a, b in sorted(cand):
        ta, tb = tracks[a], tracks[b]
        common = sorted(set(idx_of[a]) & set(idx_of[b]))
        if len(common) <= min_hist:
            continue
        rmap, cmap, amap = {}, {}, {}
        for f in common:
            ia_, ib_ = idx_of[a][f], idx_of[b][f]
            ba, bb = ta["box"][ia_], tb["box"][ib_]
            if _box_cover(ba, bb) < 0.12 and _box_cover(bb, ba) < 0.12:
                continue
            pa, pb = ta["poly"][ia_], tb["poly"][ib_]
            inter, a1, a2 = _poly_overlap(pa, pb)
            m = min(a1, a2)
            rmap[f] = inter / m if m > 1e-6 else 0.0
            amap[f] = m
            if pa is not None and pb is not None and len(pa) >= 3 and len(pb) >= 3:
                # 质心距 / 掩膜尺度 —— 判"是不是同一个东西"
                cmap[f] = float(np.linalg.norm(pa.mean(0) - pb.mean(0))) / \
                    max(float(np.sqrt(m)), 1e-6)
        if not rmap:
            continue
        ratios = np.asarray([rmap.get(f, 0.0) for f in common], np.float64)
        for k, f in enumerate(common):
            r = float(ratios[k])
            if r < jump:
                continue
            if cmap.get(f, 9.9) > center_ratio:
                continue                      # 质心差得远 -> 是两个真实目标
            hist = ratios[max(0, k - window):k]
            if len(hist) < min_hist:
                continue
            base = float(np.median(hist))
            if r - base < base_thr:
                continue                      # 一直是这么叠的 -> 不算突发
            ia_, ib_ = idx_of[a][f], idx_of[b][f]
            ca, cb = float(ta["conf"][ia_]), float(tb["conf"][ib_])
            if (ca < drop_conf) != (cb < drop_conf):
                victim = a if ca < drop_conf else b
                why = "低分"
            elif ca != cb:
                victim = a if ca < cb else b
                why = "低分"
            else:
                victim = a if amap.get(f, 0.0) <= amap.get(f, 1.0) else b
                why = "面积小"
            drops.setdefault(victim, []).append(idx_of[victim][f])
            other = b if victim == a else a
            events.append({"f": f, "cls": CLS_NAMES[tracks[victim]["cls"]],
                           "num": tracks[victim]["num"], "r": r, "base": base,
                           "conf": ca if victim == a else cb, "why": why,
                           "kcls": CLS_NAMES[tracks[other]["cls"]],
                           "knum": tracks[other]["num"]})

    dead = []
    for tid, pos in drops.items():
        _drop_positions(tracks[tid], pos)
        if not tracks[tid]["fi"]:
            dead.append(tid)
    for tid in dead:
        tracks.pop(tid)

    if verbose and events:
        print(f"[fix-overlap] 重复掩膜重算 {len(events)} 帧"
              f"（该帧由插值重建）", flush=True)
        for e in events[:8]:
            print(f"         {e['cls']:11s} #{e['num']:<3d} f{e['f']:<5d} "
                  f"重叠 {e['r']*100:4.0f}% (历史中位 {e['base']*100:4.0f}%) "
                  f"conf {e['conf']:.2f} [{e['why']}] -> 丢弃，保留 "
                  f"{e['kcls']} #{e['knum']}", flush=True)
    return len(events)


# ----------------------------------------------------------------------------
# 平滑：短间隙插值 + Savitzky-Golay 时间域滤波
# ----------------------------------------------------------------------------
_SG_CACHE = {}


def _sg_coeffs(win, order):
    """Savitzky-Golay 中心平滑系数（最小二乘等价实现，不依赖 scipy）。"""
    key = (win, order)
    if key in _SG_CACHE:
        return _SG_CACHE[key]
    half = win // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, order + 1, increasing=True)
    e = np.zeros(order + 1)
    e[0] = 1.0
    c = e @ np.linalg.pinv(A.T @ A) @ A.T
    _SG_CACHE[key] = c
    return c


def _savgol(y, win, order=2):
    """对一维序列做 Savitzky-Golay 平滑，窗口自动收缩，边界用镜像填充。"""
    y = np.asarray(y, np.float64)
    n = len(y)
    if n < 3:
        return y.copy()
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 3:
        return y.copy()
    if win % 2 == 0:
        win -= 1
    order = min(order, win - 1)
    if order < 1:
        order = 1
    c = _sg_coeffs(win, order)
    half = win // 2
    yp = np.concatenate([y[half:0:-1], y, y[-2:-half - 2:-1]])
    if len(yp) < win:
        return y.copy()
    return np.convolve(yp, c[::-1], mode="valid")


def smooth_tracks(tracks, nframes, win=15, max_gap=20, poly_win=11,
                  poly_outlier=2.2, shape_max=0.06, clamp_tol=0.03,
                  poly_fill_gap=20, fill_max_body=2.5):
    """就地生成平滑后的逐帧结果。

    返回 {frame_idx: [ {tid, cls, num, conf, box, poly} ]}

    `poly_fill_gap`：同一编号在中间缺帧时，缺口 <= 该帧数就用插值补出掩膜。
    `fill_max_body`：缺口两端中心距不得超过多少个目标尺度，否则不补。
    """
    out = {}
    n_fill = 0
    n_fill_tracks = 0
    fill_log = []
    for tid, tr in tracks.items():
        _before = n_fill
        fi = np.asarray(tr["fi"], np.int32)
        box = np.asarray(tr["box"], np.float32)
        cf = np.asarray(tr["conf"], np.float32)
        if len(fi) == 0:
            continue

        cx = (box[:, 0] + box[:, 2]) / 2
        cy = (box[:, 1] + box[:, 3]) / 2
        bw = box[:, 2] - box[:, 0]
        bh = box[:, 3] - box[:, 1]

        # 完整时间轴 + 线性插值补短间隙
        f0, f1 = int(fi[0]), int(fi[-1])
        axis = np.arange(f0, f1 + 1)
        has = np.zeros(len(axis), bool)
        has[fi - f0] = True

        def fill(v):
            return np.interp(axis, fi, v)

        vals = [fill(cx), fill(cy), fill(bw), fill(bh), fill(cf)]

        # 只对「连续出现段」做平滑，gap 过长则分段（避免跨越长缺失插值出假目标）
        segs = []
        s = 0
        for i in range(1, len(axis) + 1):
            if i == len(axis) or not has[i]:
                segs.append((s, i))
                s = i + 1
        # 把短的 gap 合并回来
        merged = []
        for seg in segs:
            seg_has = has[seg[0]:seg[1]]
            if not seg_has.any():
                continue
            if merged and seg[0] - merged[-1][1] <= max_gap:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(list(seg))

        sm = [np.array(v, np.float64) for v in vals]
        for a, bb in merged:
            if bb - a < 3:
                continue
            for v in sm:
                v[a:bb] = _savgol(v[a:bb], win)

        # ------------------------------------------------------------------
        # 轮廓：框归一化 -> 时序对齐 -> 平滑 -> 形状限速 -> 收边 -> 反投影
        # ------------------------------------------------------------------
        polys = tr["poly"]
        have_poly = [i for i, p in enumerate(polys)
                     if p is not None and len(p) == CONTOUR_POINTS]
        poly_map = {}
        if len(have_poly) >= 2:
            qs = np.stack([_box_norm_poly(polys[i], box[i]) for i in have_poly], 0)
            pf = fi[have_poly]

            # (a) 剔除面积异常帧（掩膜被遮挡/闪断导致的孤立骤缩或骤胀）
            keep = np.ones(len(qs), bool)
            if poly_outlier > 1.0 and len(qs) >= 5:
                ar = np.array([_poly_area(q) for q in qs])
                rm = _roll_med(ar, 15)
                ok = (ar > rm / poly_outlier) & (ar < rm * poly_outlier)
                if ok.sum() >= 5 and ok.sum() < len(ok):
                    keep = ok

            # (b) 循环位移对齐 —— 保证「第 i 个点」帧间对应同一物理位置
            qs = _align_stack(qs[keep])
            pf = pf[keep]

            # (c) 逐点在时间轴插值后做 Savitzky-Golay 平滑
            T, Np = len(axis), CONTOUR_POINTS
            qx = np.zeros((T, Np))
            qy = np.zeros((T, Np))
            for c in range(Np):
                qx[:, c] = np.interp(axis, pf, qs[:, c, 0])
                qy[:, c] = np.interp(axis, pf, qs[:, c, 1])
            for a, bb in merged:
                if bb - a < 3:
                    continue
                for c in range(Np):
                    qx[a:bb, c] = _savgol(qx[a:bb, c], poly_win)
                    qy[a:bb, c] = _savgol(qy[a:bb, c], poly_win)

            # (d) 形状变化限速：连续帧轮廓点位移不超过 shape_max
            qx, qy = _limit_shape(qx, qy, shape_max)

            # (e) 收进框内（允许 tol 比例的溢出），再按平滑后的框反投影
            #
            # ★ 缺帧填补：同一编号在中间有几帧没识别到、但前后都在同一位置附近，
            #   这里用线性插值（框）+ 轮廓插值（形状）把那几帧补出来。
            #
            #   为什么必须做：原来 `if not has[i]: continue` 只给**真实存在的帧**
            #   生成轮廓，插值只补了框参数、没补掩膜，于是缺帧在成片里是"对象凭空
            #   消失"（f2603 就一直是这样）。而且这个 `continue` 在**两处**
            #   （生成 poly_map 处 + 后面的逐帧装配处），两处都要放行。
            #
            #   填补的三个前提（防止乱补）：
            #     a. 缺口**在中间**：前后都要有真实观测（首尾不算，否则会往前/后
            #        凭空延长轨迹）；
            #     b. 缺口长度 <= `poly_fill_gap` 帧；
            #     c. **位置相近**：缺口两端的中心距 <= `fill_max_body` 个目标尺度
            #        （用 sqrt(宽×高) 作尺度）。距离过大说明中间可能发生了 ID 跳变
            #        或对象已经走远，插值会画出"穿越墙体的残影"，宁可不补。
            own_prev = np.full(T, -1, np.int64)
            own_next = np.full(T, -1, np.int64)
            last = -1
            for i in range(T):
                if has[i]:
                    last = i
                own_prev[i] = last
            last = -1
            for i in range(T - 1, -1, -1):
                if has[i]:
                    last = i
                own_next[i] = last
            for i in range(T):
                if not has[i]:
                    p, nx = int(own_prev[i]), int(own_next[i])
                    if p < 0 or nx < 0:
                        continue                      # 轨迹首尾，不补
                    if (nx - p - 1) > poly_fill_gap:
                        continue                      # 缺口太长
                    dist = float(np.hypot(sm[0][nx] - sm[0][p],
                                          sm[1][nx] - sm[1][p]))
                    scale = float(np.sqrt(max(sm[2][p] * sm[3][p], 1.0)))
                    if dist > fill_max_body * scale:
                        continue                      # 两端离太远，不像同一个对象
                    n_fill += 1
                    if len(fill_log) < 12 and i == p + 1:   # 每个缺口只记一次
                        fill_log.append((int(tr["num"]), CLS_NAMES[tr["cls"]],
                                         int(axis[p]), int(axis[nx]),
                                         nx - p - 1, round(dist, 1),
                                         round(dist / max(scale, 1e-6), 2)))
                q = _clamp_q(np.stack([qx[i], qy[i]], 1), clamp_tol)
                bws = max(sm[2][i], 1.0)
                bhs = max(sm[3][i], 1.0)
                poly_map[int(axis[i])] = np.stack(
                    [sm[0][i] + q[:, 0] * bws,
                     sm[1][i] + q[:, 1] * bhs], 1).astype(np.float32)
        elif len(have_poly) == 1:
            # 注意: have_poly 里存的是 polys 列表的**局部下标**, 而 has/poly_map
            # 用的是**绝对帧号**。混用会让 i-f0 越界(轨迹局部下标 < f0 时)。
            i = have_poly[0]
            j = int(fi[i]) - f0
            if 0 <= j < len(has) and has[j]:
                q = _clamp_q(_box_norm_poly(polys[i], box[i]), clamp_tol)
                poly_map[int(fi[i])] = np.stack(
                    [sm[0][j] + q[:, 0] * max(sm[2][j], 1.0),
                     sm[1][j] + q[:, 1] * max(sm[3][j], 1.0)],
                    1).astype(np.float32)

        for i, f in enumerate(axis):
            f = int(f)
            # ★ 缺帧但已用插值轮廓重建了掩膜的，也要输出（否则重建白做，
            #   被伪影清理丢掉的帧在成片里仍是"对象凭空消失"）。
            if not has[i] and f not in poly_map:
                continue
            A = sm[0][i] - sm[2][i] / 2
            B = sm[1][i] - sm[3][i] / 2
            C = sm[0][i] + sm[2][i] / 2
            D = sm[1][i] + sm[3][i] / 2
            out.setdefault(f, []).append({
                "tid": tid, "cls": tr["cls"], "num": tr["num"],
                "conf": float(sm[4][i]),
                "box": (float(A), float(B), float(C), float(D)),
                "poly": poly_map.get(f),
            })
        if n_fill > _before:
            n_fill_tracks += 1

    if n_fill:
        print(f"[fill-gap] 同一编号中间缺帧已滤波补齐：{n_fill} 帧，"
              f"涉及 {n_fill_tracks} 条轨迹"
              f"（缺口 <={poly_fill_gap} 帧且两端中心距 <={fill_max_body}×目标尺度）",
              flush=True)
        for num, cls, f0_, f1_, gap, dist, rel in fill_log:
            print(f"         {cls:11s} #{num:<3d} 补 f{f0_ + 1}~f{f1_ - 1}"
                  f"（{gap} 帧）  两端距 {dist:.0f}px = {rel:.2f}×尺度", flush=True)
    return out


# ----------------------------------------------------------------------------
# Pass 2：渲染 + 编码
# ----------------------------------------------------------------------------
def _label_rect(text, x, y, tw, tht, base, W, H):
    """把标签锚点钳制进画布，返回 (x, y, rect)。rect = (x0,y0,x1,y1)。"""
    x = max(3, min(int(x), W - tw - 4))
    y = max(tht + 4, min(int(y), H - base - 4))
    return x, y, (x - 3, y - tht - 3, x + tw + 3, y + base + 1)


def render_frame(frame, items, args, y_off=0):
    """在（可能已裁剪的）画布上绘制。y_off 为原图坐标到画布坐标的平移量。"""
    H, W = frame.shape[:2]
    polys = []
    for it in items:
        p = it["poly"]
        if p is None or len(p) < 3:
            continue
        q = p if y_off == 0 else p - np.array([0, y_off], np.float32)
        polys.append((np.round(q).astype(np.int32).reshape(-1, 1, 2),
                      CLS_COLORS[it["cls"]]))

    # ---- 掩膜：只在所有目标的联合包围盒内做混合，避免整帧运算 ----
    if polys and args.mask_alpha > 0:
        x1 = min(int(p[:, 0, 0].min()) for p, _ in polys)
        y1 = min(int(p[:, 0, 1].min()) for p, _ in polys)
        x2 = max(int(p[:, 0, 0].max()) for p, _ in polys)
        y2 = max(int(p[:, 0, 1].max()) for p, _ in polys)
        x1, y1 = max(0, x1 - 2), max(0, y1 - 2)
        x2, y2 = min(W, x2 + 3), min(H, y2 + 3)
        if x2 - x1 > 1 and y2 - y1 > 1:
            roi = frame[y1:y2, x1:x2]
            roi_over = roi.copy()
            for pts, col in polys:
                cv2.fillPoly(roi_over, [pts - np.array([x1, y1], np.int32)], col)
            cv2.addWeighted(roi_over, args.mask_alpha, roi,
                            1 - args.mask_alpha, 0, roi)

    for pts, col in polys:
        cv2.polylines(frame, [pts], True, col, args.line_width, cv2.LINE_AA)

    # ---- 标签：固定锚定在对象正上方，不左右/上下跳 ----
    # 只画「编号 + 置信度」，类别靠掩码框与填充颜色区分（不写类别文字）。
    fs, thick = args.font_scale, args.text_thick
    todo = []
    for it in items:
        A, B, C, D = it["box"]
        B -= y_off
        D -= y_off
        text = f"{it['num']} {it['conf']:.2f}"
        (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        todo.append((it, text, (A, B, C, D), tw, tht, base))

    # 从下往上画：下方对象先占位，避免上方标签盖住下方
    todo.sort(key=lambda r: -r[2][3])
    for it, text, box, tw, tht, base in todo:
        A, B, C, D = box
        # ★ 固定方位：水平居中于对象，紧贴对象正上方
        x = (A + C) / 2.0 - tw / 2.0
        y = B - 5
        px, py, rect = _label_rect(text, x, y, tw, tht, base, W, H)
        col = CLS_COLORS[it["cls"]]
        if args.draw_box:
            cv2.rectangle(frame, (int(A), int(B)), (int(C), int(D)), col,
                          max(1, args.line_width - 1), cv2.LINE_AA)
        if args.label_bg:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), col, -1)
            cv2.putText(frame, text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), thick, cv2.LINE_AA)
        else:
            # 无底色：先用类别色描边再叠加白色，保证在任何背景上都可读
            cv2.putText(frame, text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        col, thick + 2, cv2.LINE_AA)
            cv2.putText(frame, text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), thick, cv2.LINE_AA)
    return frame


def run(args):
    if args.quiet:
        import warnings
        import logging
        warnings.filterwarnings("ignore")
        logging.disable(logging.WARNING)
    src = args.source
    if not os.path.isfile(src):
        raise SystemExit(f"source not found: {src}")
    if args.out is None:
        os.makedirs(args.out_dir, exist_ok=True)
        args.out = os.path.join(
            args.out_dir, os.path.splitext(os.path.basename(src))[0] + "_track.mp4")

    print("=" * 72)
    print(f"source : {src}")
    print(f"output : {args.out}")
    print(f"model  : {os.path.basename(args.model)}")
    print(f"imgsz  : {args.imgsz} | conf: {args.conf} (首现 {args.conf_init}) | "
          f"tracker: {args.tracker}")
    print(f"order  : {args.order} | smooth: {args.smooth_win} | "
          f"dedup: {args.dedup_ratio} | mask a: {args.mask_alpha} | "
          f"矩形框: {'画' if args.draw_box else '不画'}")
    # 明确报告推理设备（--device 以前没真正传给模型，导致"看起来像在跑 CPU"）
    try:
        import torch
        if str(args.device).lower() == "cpu":
            print("device : CPU（按参数指定）")
        elif torch.cuda.is_available():
            i = int(args.device) if str(args.device).isdigit() else 0
            print(f"device : cuda:{i} — {torch.cuda.get_device_name(i)}")
        else:
            print("device : CPU（未检测到可用 CUDA）")
    except Exception as e:
        print(f"device : {args.device}（无法探测: {e}）")
    print("=" * 72, flush=True)

    from ultralytics import YOLO

    # ---------- Pass 1 ----------
    # ver 4: 去重加「已建轨迹近乎完全重合才去重」(dedup_dup) 判据
    sig = {"ver": 5, "source": src, "model": os.path.basename(args.model),
           "imgsz": args.imgsz, "conf": args.conf, "tracker": args.tracker,
           "limit": args.limit, "cls_tol": args.cls_tol,
           "dedup_ratio": args.dedup_ratio, "dedup_cross": args.dedup_cross,
           "dedup_dup": args.dedup_dup, "conf_init": args.conf_init}
    cache = args.cache or os.path.splitext(args.out)[0] + "_pass1.pkl"
    if args.use_cache and os.path.isfile(cache):
        import pickle
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        if saved.get("sig") == sig:
            tracks, st, meta = saved["tracks"], saved["stats"], saved["meta"]
            print(f"[pass 1] 命中缓存 {os.path.basename(cache)}", flush=True)
        else:
            print("[pass 1] 缓存参数不匹配，重新推理", flush=True)
            saved = None
    else:
        saved = None

    if saved is None:
        print("[pass 1] 检测 + 跟踪", flush=True)
        model = YOLO(args.model)
        tracks, st, meta = pass1_collect(
            model, src, args.imgsz, args.conf, args.tracker, args.limit,
            args.dedup_ratio, args.order, half=args.half,
            dedup_cross=args.dedup_cross, cls_tol=args.cls_tol,
            dedup_dup=args.dedup_dup, conf_init=args.conf_init,
            device=args.device)
        if args.use_cache:
            import pickle
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump({"sig": sig, "tracks": tracks, "stats": st,
                             "meta": meta}, f)
            print(f"[pass 1] 缓存 -> {os.path.basename(cache)}", flush=True)

    W, H, fps, total, eff_top, eff_bot = meta
    todo = st["frames"]
    border = (args.border_margin, eff_top + args.border_margin,
              W - args.border_margin, eff_bot - args.border_margin)
    print(f"         画面有效行 y∈[{eff_top},{eff_bot}]  "
          f"边界带 x<{border[0]} / x>{border[2]} / y<{border[1]} / y>{border[3]}",
          flush=True)
    print(f"[pass 1] 完成 {todo} 帧 / {st['elapsed']:.0f}s "
          f"({todo/max(st['elapsed'],1e-9):.1f} fps)  张量设备={st.get('device','n/a')}",
          flush=True)
    print(f"         原始实例 {st['raw']} | 去重丢弃 {st['dedup']}"
          + (f"（其中跨类重复 {st.get('dedup_cross', 0)}）" if st.get("dedup_cross") else "")
          + f" | 轨迹 {st['new_track']}", flush=True)
    if st.get("dedup_keep_est") or st.get("cls_tol"):
        print(f"         去重保住已建轨迹连续帧 {st.get('dedup_keep_est', 0)} 次"
              f" | 类别跳变容忍 {st.get('cls_tol', 0)} 帧", flush=True)

    # ---------- 轨迹缝合（消除对象短暂丢失导致的编号跳变） ----------
    # 注意顺序：**先缝合，再过滤短轨迹**。
    # 反过来的话，一条 3 帧的碎片会在缝合前就被删掉 —— 它本来能把前后两段长
    # 轨迹桥接起来，删掉等于把缺口从 3 帧硬生生撑大到几十帧，反而更难缝合。
    if args.stitch:
        tracks, merged = stitch_tracks(
            tracks, max_gap=args.stitch_gap, max_res=args.stitch_res,
            max_angle=args.stitch_angle, border=border)
        if not merged:
            print("[stitch] 无需缝合", flush=True)
    else:
        print("[stitch] 已关闭", flush=True)

    # ---------- 掩膜伪影清理（在平滑之前做，坏帧才能被插值重建） ----------
    if args.fix_merge:
        fix_merged_masks(tracks, area_ratio=args.merge_area_ratio,
                         cover=args.merge_cover, look=args.merge_look,
                         max_len=args.merge_max_len)
    if args.fix_overlap:
        fix_overlap_conflicts(tracks, jump=args.overlap_jump,
                              base_thr=args.overlap_base,
                              drop_conf=args.overlap_drop_conf,
                              center_ratio=args.overlap_center)

    # 过滤过短轨迹（多为误检）—— 放在缝合之后
    if args.min_track_frames > 1:
        drop = [t for t, v in tracks.items() if len(v["fi"]) < args.min_track_frames]
        for t in drop:
            tracks.pop(t)
        if drop:
            print(f"         过滤短轨迹 {len(drop)} 条 (<{args.min_track_frames} 帧)", flush=True)

    # ---------- 平滑 ----------
    print(f"[smooth] 插值 + Savitzky-Golay (win={args.smooth_win}, "
          f"poly={args.poly_win}) | 形状限速 {args.shape_max} | "
          f"掩膜收边 {args.clamp_tol}", flush=True)
    t0 = time.time()
    per_frame = smooth_tracks(tracks, todo, win=args.smooth_win,
                              max_gap=args.max_gap, poly_win=args.poly_win,
                              poly_outlier=args.poly_outlier,
                              shape_max=args.shape_max,
                              clamp_tol=args.clamp_tol,
                              poly_fill_gap=args.poly_fill_gap,
                              fill_max_body=args.fill_max_body)
    print(f"[smooth] 完成 {time.time()-t0:.1f}s", flush=True)

    # ---------- 统计 ----------
    per_cls = {}
    for tr in tracks.values():
        d = per_cls.setdefault(tr["cls"], {"tracks": 0, "nums": []})
        d["tracks"] += 1
        d["nums"].append(tr["num"])
    print("-" * 72)
    for c in sorted(per_cls):
        d = per_cls[c]
        print(f"  {CLS_NAMES[c]:11s} 轨迹 {d['tracks']:4d} 条, "
              f"编号 1~{max(d['nums']) if d['nums'] else 0}")
    cnt = [len(v) for v in per_frame.values()]
    if cnt:
        print(f"  每帧目标数: 平均 {np.mean(cnt):.1f}  最多 {max(cnt)}")
    print("-" * 72, flush=True)

    # ---------- Pass 2 ----------
    if args.crop:
        cy0 = max(0, eff_top - args.crop_pad)
        cy1 = min(H, eff_bot + 1 + args.crop_pad)
    else:
        cy0, cy1 = 0, H
    # H.264 的 yuv420p 要求宽高均为偶数，否则 libx264 拒绝编码
    # （表现为 ffmpeg 立即退出、管道写入报 "Invalid argument"）
    if (cy1 - cy0) % 2:
        cy1 -= 1
    OH = cy1 - cy0
    print(f"[pass 2] 渲染 + 编码  (输出 {W}x{OH}"
          + (f"，已裁掉上下黑边 y<{cy0} / y>{cy1-1}" if args.crop else "，保留完整画面")
          + ")", flush=True)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cmd = [ff, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{W}x{OH}", "-pix_fmt", "bgr24", "-r", f"{fps}", "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    errlog_path = os.path.splitext(args.out)[0] + ".ffmpeg.log"
    errlog = open(errlog_path, "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=errlog)

    cap = cv2.VideoCapture(src)
    t0 = time.time()
    n = 0
    while n < todo:
        ok, frame = cap.read()
        if not ok:
            break
        items = per_frame.get(n, [])
        canvas = frame if not args.crop else frame[cy0:cy1].copy()
        if canvas.shape[0] != OH:
            canvas = np.ascontiguousarray(canvas[:OH])
        render_frame(canvas, items, args, y_off=cy0)
        try:
            proc.stdin.write(canvas.tobytes())
        except (BrokenPipeError, OSError) as e:
            print(f"    编码中断（第 {n} 帧）: {e}", flush=True)
            print(f"    原因见 {errlog_path}", flush=True)
            break
        n += 1
        if n % 500 == 0 or n == todo:
            el = time.time() - t0
            print(f"    {n}/{todo} | {el:.0f}s | {n/el:.1f} fps | "
                  f"eta {(todo-n)/max(n/el,1e-9):.0f}s", flush=True)
    cap.release()
    proc.stdin.close()
    proc.wait()
    errlog.close()

    print("-" * 72)
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        print(f"done  : {args.out}  ({os.path.getsize(args.out)/1024**2:.1f} MB)")
    else:
        print(f"编码失败: {args.out}")
        try:
            with open(errlog_path, "rb") as f:
                tail = f.read()[-1500:].decode("utf-8", "replace")
            print("ffmpeg 输出尾部:")
            print(tail)
        except Exception:
            pass
    print(f"ffmpeg rc = {proc.returncode}")

    if args.dump_json:
        jp = os.path.splitext(args.out)[0] + "_tracks.json"
        payload = {
            "source": src, "model": os.path.basename(args.model),
            "imgsz": args.imgsz, "conf": args.conf,
            "conf_init": args.conf_init, "order": args.order,
            "fps": fps, "size": [W, H], "frames": todo,
            "effective_rows": [eff_top, eff_bot],
            "output_crop_y": [cy0, cy1],
            "border": list(border),
            "classes": {str(k): v for k, v in CLS_NAMES.items()},
            "tracks": [
                {"tid": t, "cls": CLS_NAMES[v["cls"]], "num": v["num"],
                 "first": int(v["fi"][0]), "last": int(v["fi"][-1]),
                 "frames": len(v["fi"]),
                 "conf_mean": round(float(np.mean(v["conf"])), 4)}
                for t, v in sorted(tracks.items(), key=lambda kv: (kv[1]["cls"], kv[1]["num"]))
            ],
        }
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"json  : {jp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model",
                    default=r"E:\AAA-reseach\runs\segment\round3_744\weights\best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.3,
                    help="跟踪门限：检测置信度下限（已建轨迹续帧用这个）")
    ap.add_argument("--conf-init", type=float, default=0.75,
                    help="首现门限：新对象建立轨迹的置信度下限，须 >= --conf")
    ap.add_argument("--tracker", default="fasttrack.yaml")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 帧，0=全部")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=r"E:\AAA-reseach\videos\output\track")
    ap.add_argument("--order", default="diag", choices=["diag", "row"],
                    help="编号空间顺序：diag=右上到左下对角, row=逐行从右到左")
    ap.add_argument("--dedup-ratio", type=float, default=0.5)
    ap.add_argument("--dedup-cross", type=float, default=0.8,
                    help="rider/pedestrian 跨类重复判定的重叠比阈值")
    ap.add_argument("--dedup-dup", type=float, default=0.9,
                    help="两条**已建**轨迹之间判为「同一物体的重复检测」所需的重叠比；"
                         "低于此值视为两个不同对象，不去重")
    ap.add_argument("--smooth-win", type=int, default=15)
    ap.add_argument("--poly-win", type=int, default=11)
    ap.add_argument("--poly-outlier", type=float, default=2.2,
                    help="掩膜面积相对滚动中位的偏离倍数，超过即视为异常帧并插值")
    ap.add_argument("--shape-max", type=float, default=0.06,
                    help="相邻帧轮廓点最大平均位移（框归一化单位），抑制形状突变；0=关闭")
    ap.add_argument("--clamp-tol", type=float, default=0.03,
                    help="掩膜允许超出检测框的比例（相对框尺寸）")
    ap.add_argument("--max-gap", type=int, default=45)
    ap.add_argument("--poly-fill-gap", type=int, default=20,
                    help="同一编号中间缺帧时，缺口<=N帧则用插值补出掩膜（0=关闭）")
    ap.add_argument("--fill-max-body", type=float, default=2.5,
                    help="补缺帧的位置合理性：缺口两端中心距<=N×目标尺度才补")
    ap.add_argument("--min-track-frames", type=int, default=5)
    ap.add_argument("--stitch", action="store_true", default=True,
                    help="缝合因短暂丢失而断裂的轨迹（保持编号）")
    ap.add_argument("--no-stitch", dest="stitch", action="store_false")
    ap.add_argument("--stitch-gap", type=int, default=40)
    ap.add_argument("--stitch-res", type=float, default=90.0)
    ap.add_argument("--stitch-angle", type=float, default=60.0,
                    help="[已弃用] 原速度方向夹角门限；方向判断已移除，此参数不再生效")
    ap.add_argument("--cls-tol", type=int, default=30,
                    help="容忍连续多少帧的类别跳变（超时才停止收集该帧）")
    # ---- 掩膜伪影清理 ----
    ap.add_argument("--fix-merge", action="store_true", default=True,
                    help="近邻小掩膜突变为一个大掩膜时丢弃该帧观测（默认开）")
    ap.add_argument("--no-fix-merge", dest="fix_merge", action="store_false")
    ap.add_argument("--merge-area-ratio", type=float, default=1.45,
                    help="掩膜面积相对前 15 帧中位的倍数，超过视为突发胀大")
    ap.add_argument("--merge-cover", type=float, default=0.5,
                    help="消失近邻的末框落在大掩膜框内的比例门限")
    ap.add_argument("--merge-look", type=int, default=3,
                    help="近邻须在本帧前多少帧内消失，才算被吞并")
    ap.add_argument("--merge-max-len", type=int, default=12,
                    help="判为「粘连大掩膜」的轨迹最长多少帧；超长视为真实目标不动")
    ap.add_argument("--fix-overlap", action="store_true", default=True,
                    help="两掩膜突发重复重叠时丢低分那个（默认开）")
    ap.add_argument("--no-fix-overlap", dest="fix_overlap", action="store_false")
    ap.add_argument("--overlap-jump", type=float, default=0.80,
                    help="判定重复掩膜的绝对重叠比门限（相对较小的掩膜）")
    ap.add_argument("--overlap-center", type=float, default=0.35,
                    help="质心距/sqrt(小掩膜面积) 上限；超过视为两个真实目标")
    ap.add_argument("--overlap-base", type=float, default=0.25,
                    help="本帧重叠比须比该对自身历史中位高出多少，才算突发")
    ap.add_argument("--overlap-drop-conf", type=float, default=0.6,
                    help="两掩膜冲突时，低于此置信度的一方优先被丢")
    ap.add_argument("--border-margin", type=int, default=60,
                    help="画面边界带宽度，落在此带内视为走出/进入画面")
    ap.add_argument("--mask-alpha", type=float, default=0.35)
    ap.add_argument("--line-width", type=int, default=3)
    ap.add_argument("--font-scale", type=float, default=0.45)
    ap.add_argument("--text-thick", type=int, default=1)
    ap.add_argument("--draw-box", dest="draw_box", action="store_true", default=False,
                    help="画出矩形检测框（默认**不画**，只画掩码多边形）")
    ap.add_argument("--no-box", dest="draw_box", action="store_false", default=False,
                    help="不画矩形检测框（默认行为，显式传亦可）")
    ap.add_argument("--label-bg", action="store_true", default=True,
                    help="编号/置信度文字带类别色底衬（默认开启）")
    ap.add_argument("--no-label-bg", dest="label_bg", action="store_false")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--crop", action="store_true", default=True,
                    help="输出时裁掉上下黑边（检测仍在完整画面上进行）")
    ap.add_argument("--no-crop", dest="crop", action="store_false")
    ap.add_argument("--crop-pad", type=int, default=22,
                    help="裁剪时上下各多留 N 像素，避免框贴边被切")
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true", default=True,
                    help="FP16 推理（GPU 上更快，数值差异极小）")
    ap.add_argument("--no-half", dest="half", action="store_false")
    ap.add_argument("--quiet", action="store_true", help="屏蔽库层警告输出")
    ap.add_argument("--dump-json", action="store_true", default=True)
    ap.add_argument("--no-json", dest="dump_json", action="store_false")
    ap.add_argument("--cache", default=None, help="pass1 缓存路径（默认 out 同名前缀 .pkl）")
    ap.add_argument("--use-cache", action="store_true", default=True)
    ap.add_argument("--no-cache", dest="use_cache", action="store_false")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
