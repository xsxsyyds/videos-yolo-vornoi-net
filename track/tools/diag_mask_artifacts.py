# -*- coding: utf-8 -*-
"""诊断 pass1 缓存里的两类掩膜伪影，为阈值选型提供实测依据。

用法：
    python tools/diag_mask_artifacts.py videos/new-track/14_track_pass1.pkl

输出：
  ① 「一个大掩膜同时包住 N 个小掩膜」的分布（N=0/1/2/3/4+ 各多少帧）
  ② 两两重叠比的分布，以及「突发跳变」的候选数与幅度
"""
import pickle
import sys

import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track_segment as T  # noqa: E402


def poly_area_or_box(tr, i):
    p = tr["poly"][i]
    if p is not None and len(p) >= 3:
        return T._poly_area(p)
    b = tr["box"][i]
    return float((b[2] - b[0]) * (b[3] - b[1])) * 0.7


def build(tracks):
    per = {}
    for tid, tr in tracks.items():
        for i, f in enumerate(tr["fi"]):
            per.setdefault(int(f), []).append((tid, i))
    idx_of = {t: {int(f): i for i, f in enumerate(tr["fi"])}
              for t, tr in tracks.items()}
    area = {t: np.asarray([poly_area_or_box(tr, i)
                           for i in range(len(tr["fi"]))], float)
            for t, tr in tracks.items()}
    return per, idx_of, area


def frame_pairs(tracks, lst, area):
    """该帧里「框基本被另一个更大框包住」的候选对 (a, b)。"""
    n = len(lst)
    if n < 2:
        return []
    bx = np.asarray([tracks[t]["box"][i] for t, i in lst], np.float32)
    ar = np.asarray([area[t][i] for t, i in lst], np.float32)
    iw = np.minimum(bx[:, None, 2], bx[None, :, 2]) - \
        np.maximum(bx[:, None, 0], bx[None, :, 0])
    ih = np.minimum(bx[:, None, 3], bx[None, :, 3]) - \
        np.maximum(bx[:, None, 1], bx[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    ba = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    rel = inter / ba[None, :].clip(1e-6)          # 相对 b 自身面积的覆盖比
    m = (rel >= 0.7) & (ar[:, None] > 1.3 * ar[None, :])
    return [(int(a), int(b)) for a, b in zip(*np.nonzero(m))]


def main():
    pkl = sys.argv[1]
    d = pickle.load(open(pkl, "rb"))
    tracks = d["tracks"]
    print(f"缓存 : {pkl}")
    print(f"签名 : conf={d['sig'].get('conf')} conf_init={d['sig'].get('conf_init')}")
    print(f"轨迹 : {len(tracks)}")
    per, idx_of, area = build(tracks)

    # ---- ① 包含关系 ----
    print("\n=== ① 一个大掩膜同时包住 N 个小掩膜（同帧共存）===")
    hist = {}
    samples = []
    for f, lst in sorted(per.items()):
        if len(lst) < 3:
            continue
        hosts_map = {}
        for a, b in frame_pairs(tracks, lst, area):
            pa = tracks[lst[a][0]]["poly"][lst[a][1]]
            pb = tracks[lst[b][0]]["poly"][lst[b][1]]
            if pa is None or pb is None:
                continue
            it, a1, a2 = T._poly_overlap(pa, pb)
            if a2 > 1e-6 and it / a2 >= 0.7:
                hosts_map.setdefault(a, []).append(b)
        for a, hs in hosts_map.items():
            k = min(len(hs), 4)
            hist[k] = hist.get(k, 0) + 1
            if len(hs) >= 2 and len(samples) < 12:
                tid, i = lst[a]
                samples.append((f, int(tracks[tid]["num"]), len(hs),
                                float(area[tid][i]), tracks[tid]["cls"]))
    for k in sorted(hist):
        print(f"  包住 {k}{'+' if k == 4 else ''} 个 : {hist[k]:6d} 次")
    print("  样例：")
    for f, num, n, a, c in samples:
        print(f"    f{f:<5d} #{num:<3d} {T.CLS_NAMES[c]:11s} 面积 {a:8.0f} 包住 {n} 个")
    print(f"  ⇒ 满足「包住 >=2 个」的实例数：{sum(v for k, v in hist.items() if k >= 2)}")

    # ---- ② 两两重叠比 & 突发跳变 ----
    print("\n=== ② 两两掩膜重叠比 / 突发跳变 ===")
    cand = set()
    for f, lst in per.items():
        if len(lst) < 2:
            continue
        bx = np.asarray([tracks[t]["box"][i] for t, i in lst], np.float32)
        ba = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
        iw = np.minimum(bx[:, None, 2], bx[None, :, 2]) - \
            np.maximum(bx[:, None, 0], bx[None, :, 0])
        ih = np.minimum(bx[:, None, 3], bx[None, :, 3]) - \
            np.maximum(bx[:, None, 1], bx[None, :, 1])
        it = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        rel = it / np.minimum(ba[:, None], ba[None, :]).clip(1e-6)
        ii, jj = np.nonzero(np.triu(rel >= 0.3, 1))
        for x, y in zip(ii, jj):
            a, b = lst[int(x)][0], lst[int(y)][0]
            cand.add((a, b) if a < b else (b, a))
    print(f"  框重叠>=0.3 的候选轨迹对：{len(cand)}")

    peak = []
    for a, b in sorted(cand):
        common = sorted(set(idx_of[a]) & set(idx_of[b]))
        if len(common) < 6:
            continue
        rmap = {}
        for f in common:
            ia_, ib_ = idx_of[a][f], idx_of[b][f]
            ba_, bb_ = tracks[a]["box"][ia_], tracks[b]["box"][ib_]
            if T._box_cover(ba_, bb_) < 0.12 and T._box_cover(bb_, ba_) < 0.12:
                continue
            it, a1, a2 = T._poly_overlap(tracks[a]["poly"][ia_],
                                         tracks[b]["poly"][ib_])
            m = min(a1, a2)
            rmap[f] = it / m if m > 1e-6 else 0.0
        if not rmap:
            continue
        ratios = np.asarray([rmap.get(f, 0.0) for f in common], float)
        for k, f in enumerate(common):
            r = float(ratios[k])
            if r < 0.45:
                continue
            hist_ = ratios[max(0, k - 15):k]
            if len(hist_) < 4:
                continue
            base = float(np.median(hist_))
            peak.append((r - base, r, base, f,
                         int(tracks[a]["num"]), int(tracks[b]["num"])))
    peak.sort(reverse=True)
    print(f"  重叠比>=0.45 且历史>=4 的帧：{len(peak)}")
    for dlt, r, base, f, na, nb in peak[:12]:
        print(f"    f{f:<5d} #{na}-#{nb}  本帧 {r*100:4.0f}%  "
              f"历史中位 {base*100:4.0f}%  跳变 {dlt*100:4.0f}pt")
    if peak:
        dl = np.asarray([p[0] for p in peak])
        for th in (0.15, 0.25, 0.35, 0.45):
            print(f"    跳变 >={th:.2f} 的帧数：{int((dl >= th).sum())}")


if __name__ == "__main__":
    main()
