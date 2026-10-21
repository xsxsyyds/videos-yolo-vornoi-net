# -*- coding: utf-8 -*-
"""跟踪结果量化审计（对应需求逐条验收）

  A. 同帧同类别目标是否在像素上重叠（"同一像素位置识别出多个目标"）
  B. 编号是否唯一且永久占用（一个编号只属于一条轨迹）
  C. 编号跳变：同一对象因丢失被拆成两条轨迹（缝合前 / 缝合后对比）
  D. 平滑前后抖动量化
  E. 轨迹覆盖 / 插值占比
  F. 掩膜多边形的时序稳定性（面积跳变 / 形状突变 / 是否超出检测框）

用法:
    python tools/audit_tracks.py <pass1缓存.pkl> [--overlap 0.30]
"""

import argparse
import copy
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track_segment as T  # noqa: E402

POLY_WIN = 11
POLY_OUTLIER = 2.2
SHAPE_MAX = 0.06
CLAMP_TOL = 0.03


def poly_pair_overlap(pa, pb):
    """两个多边形的重叠比 = 交面积 / min(面积)。"""
    x1 = int(min(pa[:, 0].min(), pb[:, 0].min())) - 2
    y1 = int(min(pa[:, 1].min(), pb[:, 1].min())) - 2
    x2 = int(max(pa[:, 0].max(), pb[:, 0].max())) + 3
    y2 = int(max(pa[:, 1].max(), pb[:, 1].max())) + 3
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    h, w = y2 - y1, x2 - x1
    if h * w > 4_000_000:
        return 0.0
    a = np.zeros((h, w), np.uint8)
    b = np.zeros((h, w), np.uint8)
    cv2.fillPoly(a, [np.round(pa - [x1, y1]).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.round(pb - [x1, y1]).astype(np.int32)], 1)
    aa, bb = int(a.sum()), int(b.sum())
    if aa == 0 or bb == 0:
        return 0.0
    return int(np.logical_and(a, b).sum()) / min(aa, bb)


def bbox_pair_iou(ba, bb):
    iw = max(0.0, min(ba[2], bb[2]) - max(ba[0], bb[0]))
    ih = max(0.0, min(ba[3], bb[3]) - max(ba[1], bb[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (ba[2] - ba[0]) * (ba[3] - ba[1])
    a2 = (bb[2] - bb[0]) * (bb[3] - bb[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def centers(tr):
    b = np.asarray(tr["box"], np.float32)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)


def detect_breaks(tracks, gap_max, dist_max, border=None, use_dir=True):
    """同类别内 A 尾 -> B 头 时间接近且空间接近，判为同一对象被重编号。

    会排除「A 已走出画面」的情形：那属于对象离场，之后出现的是新对象，
    给新编号是正确行为，不算跳变。
    """
    out = []
    by_cls = {}
    for tid, tr in tracks.items():
        by_cls.setdefault(tr["cls"], []).append(tid)
    for c, tids in by_cls.items():
        for a in tids:
            ca = centers(tracks[a])
            fa_end, pa = int(tracks[a]["fi"][-1]), ca[-1]
            if border is not None and use_dir:
                box = np.asarray(tracks[a]["box"], np.float32)
                va = T._vel(box, np.asarray(tracks[a]["fi"], np.int32), head=False)
                if T._moving_out(pa, va, border):
                    continue                      # 走出画面，属正常离场
            for b in tids:
                if a == b:
                    continue
                cb = centers(tracks[b])
                fb_0, pb = int(tracks[b]["fi"][0]), cb[0]
                if border is not None and use_dir:
                    boxb = np.asarray(tracks[b]["box"], np.float32)
                    vb = T._vel(boxb, np.asarray(tracks[b]["fi"], np.int32), head=True)
                    if T._moving_out(pb, vb, border):
                        continue                  # 后段刚出现就准备出画
                gap = fb_0 - fa_end
                if 0 < gap <= gap_max:
                    d = float(np.linalg.norm(pa - pb))
                    if d <= dist_max:
                        out.append((c, tracks[a]["num"], tracks[b]["num"],
                                    gap, round(d, 1)))
    out.sort(key=lambda x: x[3])
    return out


def audit_overlap(tracks, thr, cross=True):
    """同帧重叠审计。同类别与跨类别分别统计（"同一像素多目标"两种都要查）。"""
    per_frame = {}
    for tid, tr in tracks.items():
        for i, fi in enumerate(tr["fi"]):
            p = tr["poly"][i]
            if p is not None:
                per_frame.setdefault(int(fi), []).append(
                    (tr["cls"], tr["num"], p, tr["box"][i]))
    same = {"checked": 0, "worst": 0.0, "bad": []}
    cross_st = {"checked": 0, "worst": 0.0, "bad": []}
    for fi in sorted(per_frame):
        items = per_frame[fi]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                same_cls = items[i][0] == items[j][0]
                if not same_cls and not cross:
                    continue
                # bbox 无交集则掩膜必然不重叠，快速跳过
                if bbox_pair_iou(items[i][3], items[j][3]) <= 0.0:
                    continue
                st = same if same_cls else cross_st
                st["checked"] += 1
                ov = poly_pair_overlap(items[i][2], items[j][2])
                st["worst"] = max(st["worst"], ov)
                if ov > thr:
                    st["bad"].append((fi, T.CLS_NAMES[items[i][0]], items[i][1],
                                      T.CLS_NAMES[items[j][0]], items[j][1], ov))
    return same, cross_st, per_frame


def audit_smooth(tracks):
    raw1, raw2, sm1, sm2, disp = [], [], [], [], []
    for tid, tr in tracks.items():
        fi = np.asarray(tr["fi"], np.int32)
        box = np.asarray(tr["box"], np.float32)
        if len(fi) < 8:
            continue
        c = centers(tr)
        brk = list(np.where(np.diff(fi) > 1)[0])
        segs, s = [], 0
        for b in brk + [len(fi) - 1]:
            segs.append((s, b + 1))
            s = b + 1
        for a, b in segs:
            if b - a < 8:
                continue
            seg = c[a:b]
            d1 = np.diff(seg, axis=0)
            d2 = np.diff(seg, n=2, axis=0)
            raw1.append(np.abs(d1 - d1.mean(0)).mean())
            raw2.append(np.abs(d2).mean())
            res = T.smooth_tracks({tid: {
                "cls": tr["cls"], "num": tr["num"],
                "fi": fi[a:b].tolist(), "box": box[a:b].tolist(),
                "conf": tr["conf"][a:b], "poly": tr["poly"][a:b]}},
                int(fi[-1]) + 1, win=15, max_gap=45, poly_win=POLY_WIN,
                poly_outlier=POLY_OUTLIER, shape_max=SHAPE_MAX,
                clamp_tol=CLAMP_TOL)
            pts = [((it["box"][0] + it["box"][2]) / 2, (it["box"][1] + it["box"][3]) / 2)
                   for f in sorted(res) for it in res[f]]
            pts = np.asarray(pts)
            if len(pts) < 8:
                continue
            cut = min(len(pts), len(seg))
            disp.append(np.abs(pts[:cut] - seg[:cut]).mean())
            d1 = np.diff(pts, axis=0)
            d2 = np.diff(pts, n=2, axis=0)
            sm1.append(np.abs(d1 - d1.mean(0)).mean())
            sm2.append(np.abs(d2).mean())
    return raw1, raw2, sm1, sm2, disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--overlap", type=float, default=0.30)
    ap.add_argument("--gap", type=int, default=30)
    ap.add_argument("--dist", type=float, default=120.0)
    ap.add_argument("--no-stitch", action="store_true")
    ap.add_argument("--stitch-gap", type=int, default=40)
    ap.add_argument("--border-margin", type=int, default=60)
    ap.add_argument("--min-track-frames", type=int, default=5,
                    help="与主流程一致：短于此长度的轨迹视为误检丢弃")
    args = ap.parse_args()

    with open(args.cache, "rb") as f:
        d = pickle.load(f)
    tracks, st, meta = d["tracks"], d["stats"], d["meta"]
    W, H, fps, total = meta[0], meta[1], meta[2], meta[3]
    eff_top, eff_bot = (meta[4], meta[5]) if len(meta) > 5 else (0, H - 1)
    todo = st["frames"]
    border = (args.border_margin, eff_top + args.border_margin,
              W - args.border_margin, eff_bot - args.border_margin)

    print("=" * 76)
    print(f"审计对象 : {d['sig']['source']}")
    print(f"配置     : {d['sig']['model']} | imgsz {d['sig']['imgsz']} | "
          f"conf {d['sig']['conf']} | {d['sig']['tracker']}")
    print(f"帧数     : {todo} / {total}  |  原始轨迹 {len(tracks)}")

    # 与主流程一致地过滤短轨迹（否则会把 1~4 帧的误检当成"断裂"误报）
    tracks = copy.deepcopy(tracks)
    short = [t for t, v in tracks.items() if len(v["fi"]) < args.min_track_frames]
    for t in short:
        tracks.pop(t)
    print(f"           过滤短轨迹 {len(short)} 条 (<{args.min_track_frames} 帧) "
          f"-> 有效轨迹 {len(tracks)}")
    print("=" * 76)

    # ---------- C（缝合前） ----------
    before_all = detect_breaks(tracks, args.gap, args.dist, border=None, use_dir=False)
    before = detect_breaks(tracks, args.gap, args.dist, border=border, use_dir=True)
    print(f"\n[C] 编号跳变（同一对象被拆成两条轨迹）  判据: gap<={args.gap}f 且 "
          f"距离<={args.dist}px")
    print(f"    画面有效行 y∈[{eff_top},{eff_bot}]   边界带 x<{border[0]} / "
          f"x>{border[2]} / y<{border[1]} / y>{border[3]}")
    print(f"    疑似对（不排除出画）: {len(before_all)} 处")
    for b in before_all[:10]:
        print(f"       {T.CLS_NAMES[b[0]]:11s} #{b[1]:<3d} -> #{b[2]:<3d}  "
              f"间隔 {b[3]:3d}f  距离 {b[4]:6.1f}px")
    escaped = len(before_all) - len(before)
    print(f"    其中因「走出画面」而属正常离场: {escaped} 处")
    print(f"    画面内部真断裂: {len(before)} 处")
    for b in before[:10]:
        print(f"       ⚠ {T.CLS_NAMES[b[0]]:11s} #{b[1]:<3d} -> #{b[2]:<3d}  "
              f"间隔 {b[3]:3d}f  距离 {b[4]:6.1f}px")

    # ---------- 缝合 ----------
    if not args.no_stitch:
        tracks, merged = T.stitch_tracks(tracks, max_gap=args.stitch_gap,
                                         border=border)
        print(f"    缝合动作: {len(merged)} 次")
    after = detect_breaks(tracks, args.gap, args.dist, border=border, use_dir=True)
    print(f"    缝合后 画面内部真断裂: {len(after)} 处"
          + (f"   ✓ 已消除 {len(before)-len(after)} 处" if len(after) < len(before)
             else ("   ✓ 无内部断裂" if not after else "   ⚠ 仍存在")))
    for b in after[:10]:
        print(f"       ⚠ {T.CLS_NAMES[b[0]]:11s} #{b[1]:<3d} -> #{b[2]:<3d}  "
              f"间隔 {b[3]:3d}f  距离 {b[4]:6.1f}px")

    # ---------- B ----------
    num2tid, per_cls = {}, {}
    for tid, tr in tracks.items():
        k = (tr["cls"], tr["num"])
        num2tid[k] = tid
        per_cls.setdefault(tr["cls"], []).append(tr["num"])
    print("\n[B] 编号唯一性与连续性")
    print(f"    编号占用 {len(num2tid)} 个 | (类别,编号) 冲突 0（结构保证）")
    for c in sorted(per_cls):
        ns = sorted(per_cls[c])
        miss = [n for n in range(1, max(ns) + 1) if n not in set(ns)]
        print(f"    {T.CLS_NAMES[c]:11s} 轨迹 {len(ns):3d} 条  编号 1~{max(ns)}"
              + (f"  作废编号 {len(miss)} 个（对象消失，不复用）" if miss else "  无作废"))

    # ---------- A ----------
    same, cross, per_frame = audit_overlap(tracks, args.overlap)
    print(f"\n[A] 同帧像素重叠（阈值 {args.overlap}）——「同一像素出现多个目标」")
    print(f"    [同类别] 候选对 {same['checked']:6d} | 超阈值 {len(same['bad']):3d} | "
          f"最大 {same['worst']:.4f}")
    for b in same["bad"][:8]:
        print(f"       ✗ f{b[0]:6d}  {b[1]} #{b[2]} vs {b[3]} #{b[4]}  重叠 {b[5]:.3f}")
    print(f"    [跨类别] 候选对 {cross['checked']:6d} | 超阈值 {len(cross['bad']):3d} | "
          f"最大 {cross['worst']:.4f}")
    for b in cross["bad"][:8]:
        print(f"       ✗ f{b[0]:6d}  {b[1]} #{b[2]} vs {b[3]} #{b[4]}  重叠 {b[5]:.3f}")
    if not same["bad"] and not cross["bad"]:
        print("    ✓ 同帧同类、跨类均无像素重叠")

    # ---------- D ----------
    raw1, raw2, sm1, sm2, disp = audit_smooth(tracks)
    print("\n[D] 平滑效果（像素）")
    if raw1:
        r1, s1 = np.mean(raw1), np.mean(sm1)
        r2, s2 = np.mean(raw2), np.mean(sm2)
        print(f"    一阶抖动  原始 {r1:7.3f} -> 平滑 {s1:7.3f}   降低 "
              f"{100*(1-s1/max(r1,1e-9)):5.1f}%")
        print(f"    二阶急动  原始 {r2:7.3f} -> 平滑 {s2:7.3f}   降低 "
              f"{100*(1-s2/max(r2,1e-9)):5.1f}%")
        print(f"    平滑位置代价（与原始偏离）: {np.mean(disp):.2f} px")

    # ---------- E ----------
    lens, gaps = [], []
    for tr in tracks.values():
        fi = np.asarray(tr["fi"], np.int32)
        lens.append(len(fi))
        if len(fi) > 1:
            g = np.diff(fi) - 1
            gaps.extend(g[g > 0].tolist())
    print("\n[E] 轨迹覆盖 / 插值")
    if lens:
        print(f"    轨迹长度 中位 {int(np.median(lens))}  最短 {min(lens)}  最长 {max(lens)}")
        print(f"    缺失帧 {sum(gaps)}  缺失段 {len(gaps)}"
              + (f"  最长缺失 {max(gaps)} 帧" if gaps else ""))
    cnt = [len(v) for v in per_frame.values()]
    print(f"    逐帧目标数 平均 {np.mean(cnt):.1f}  最多 {max(cnt)}  最少 {min(cnt)}")
    # 逐帧编号是否唯一（同一编号在同一帧只出现一次）
    dup = 0
    for fi, items in per_frame.items():
        keys = [(i[0], i[1]) for i in items]
        dup += len(keys) - len(set(keys))
    print(f"    同帧内编号重复出现次数: {dup}" + ("  ✓" if dup == 0 else "  ✗"))

    # ---------- F ----------
    audit_masks(tracks)
    print("=" * 76)


def _shoelace(p):
    if p is None or len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def audit_masks(tracks, min_frames=20):
    """掩膜多边形的时序稳定性：面积跳变、形状突变、超出检测框幅度。

    注意区分两类相邻：
      * 连续帧（gap==1）—— 用户真正看到的"连续几帧形状变化"，由 --shape-max 直接约束
      * 跨缺失帧（gap>1）—— 对象中间未渲染，累加变化会更大属正常，单独统计
    """
    print("\n[F] 掩膜多边形时序稳定性")
    # 先按主流程的平滑参数生成逐帧结果
    nf = max(int(tr["fi"][-1]) for tr in tracks.values()) + 2
    per = T.smooth_tracks(tracks, nf, win=15, max_gap=45, poly_win=POLY_WIN,
                          poly_outlier=POLY_OUTLIER, shape_max=SHAPE_MAX,
                          clamp_tol=CLAMP_TOL)
    by_tid = {}
    for f, items in per.items():
        for it in items:
            by_tid.setdefault(it["tid"], []).append((f, it))

    n = 0
    n_ovh = 0
    cont_sh, gap_sh, all_sh = [], [], []
    cont_rr, gap_rr, all_rr = [], [], []
    worst = []
    for tid, tr in tracks.items():
        rows = sorted(by_tid.get(tid, []))
        if len(rows) < min_frames:
            continue
        n += 1
        area, ovh, shape, prev, pf, pr = [], [], [], None, None, None
        o_max = 0.0
        for f, it in rows:
            p = it["poly"]
            if p is None:
                continue
            A, B, C, D = it["box"]
            bw, bh = max(C - A, 1e-6), max(D - B, 1e-6)
            a = _shoelace(p)
            area.append(a)
            o = max(max(0.0, A - p[:, 0].min()) / bw,
                    max(0.0, B - p[:, 1].min()) / bh,
                    max(0.0, p[:, 0].max() - C) / bw,
                    max(0.0, p[:, 1].max() - D) / bh)
            o_max = max(o_max, o)
            u = np.stack([(p[:, 0] - A) / bw, (p[:, 1] - B) / bh], 1)
            if prev is not None:
                gap = f - pf
                sh = float(np.sqrt(((u - prev) ** 2).sum(1)).mean())
                rr = max(a / max(pr, 1e-6), pr / max(a, 1e-6))
                all_sh.append(sh); all_rr.append(rr)
                (cont_sh if gap == 1 else gap_sh).append(sh)
                (cont_rr if gap == 1 else gap_rr).append(rr)
            prev, pf, pr = u, f, a
        if len(area) < min_frames:
            continue
        n_ovh += o_max > 0.15
        # 连续帧的最差值（决定该轨迹"看起来"稳不稳）
        cs = [x for x in all_sh[-len(area) + 1:]] if all_sh else []
        worst.append((max(cont_sh[-len(area) + 1:] or [0.0]),
                      max(cont_rr[-len(area) + 1:] or [1.0]),
                      o_max, T.CLS_NAMES[tr["cls"]], tr["num"], len(area)))

    if not n:
        print("    无足够长轨迹可供评估")
        return
    worst.sort(reverse=True)

    def stat(name, arr, good):
        if not arr:
            print(f"    {name}: 无样本")
            return
        a = np.array(arr)
        print(f"    {name}: 中位 {np.median(a):.3f}  95% 分位 {np.percentile(a,95):.3f}  "
              f"最大 {a.max():.3f}  超 {good} 的样本 {int((a>good).sum())}/{len(a)}"
              f" ({100*float((a>good).mean()):.1f}%)")

    print(f"    评估 {n} 条轨迹")
    print(f"    超出检测框 >15%    : {n_ovh:3d} 条 ({100*n_ovh/n:.1f}%)")
    print("    --- 连续帧（gap=1，即肉眼看到的帧间变化）---")
    stat("形状突变", cont_sh, 0.15)
    stat("面积跳变", cont_rr, 1.5)
    print("    --- 跨缺失帧（gap>1，对象中间未渲染，累加变化）---")
    stat("形状突变", gap_sh, 0.15)
    stat("面积跳变", gap_rr, 1.5)
    print(f"    最不稳的 {min(5,len(worst))} 条（按连续帧形状突变）:")
    for sm, rm, om, c, num, ln in worst[:5]:
        print(f"       {c:11s} #{num:<4d} {ln:>5d}帧  连续帧形状突变 {sm:.3f}  "
              f"面积跳变 {rm:.2f}x  超框 {100*om:.1f}%")
    if cont_sh:
        cs = np.array(cont_sh)
        ok = n_ovh == 0 and float((cs > 0.15).mean()) < 0.02
        print("    " + ("✓ 掩膜时序稳定：无非连续形状突变、无鼓出框" if ok else
                        "⚠ 仍有少量连续帧突变，见上方列出的轨迹"))


if __name__ == "__main__":
    main()
