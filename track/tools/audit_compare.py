# -*- coding: utf-8 -*-
"""多跟踪器审计对比：对若干 pass1 缓存分别跑 audit_tracks，输出并列指标表。

与 compare_trackers.py 的区别：
  compare_trackers.py  —— 现场跑模型+跟踪（慢，需 GPU，只给 ID 稳定性粗指标）
  audit_compare.py     —— 复用已有的 pass1 缓存，跑完整六维审计（快，指标全面）
                          适合在已经跑完多组跟踪后做最终裁决。

用法:
    python tools/audit_compare.py 名字=cache.pkl [名字=cache.pkl ...]

例:
    python tools/audit_compare.py \
        fasttrack=.../r4_9_track_pass1.pkl \
        shape=.../shape_9_track_pass1.pkl
"""
import argparse
import io
import os
import re
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audit_tracks as A  # noqa: E402


# 从审计文本里抓关键数字（容忍不同措辞）
PATTERNS = {
    "tracks": r"(?:原始轨迹|轨迹条数|轨迹数)[：:\s]*(\d+)",
    "emitted": r"编号占用[：:\s]*(\d+)",
    "broken": r"真断裂[：:\s]*(\d+)",
    "stitched": r"缝合(?:动作|段|次数)[：:\s]*(\d+)",
    "missing_seg": r"(?:缺失段|缺失片段)[：:\s]*(\d+)",
    "missing_max": r"最长缺失[：:\s]*(\d+)",
    "overlap_bad": r"超阈值[：:\s]*(\d+)",
    "area_jump": r"面积跳变[：:\s]*(\d+)",
    "shape_burst": r"形状突变[：:\s]*(\d+)",
}


def run_audit(cache_path, extra_args):
    """跑审计并捕获它的文本输出。"""
    argv_backup = sys.argv[:]
    sys.argv = ["audit_tracks.py", cache_path] + extra_args
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            A.main()
    except SystemExit:
        pass
    finally:
        sys.argv = argv_backup
    return buf.getvalue()


def parse(text):
    """按正则抽取指标；缺失的记为 None。"""
    out = {}
    for k, pat in PATTERNS.items():
        m = re.search(pat, text)
        out[k] = int(m.group(1)) if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="名字=cache.pkl")
    ap.add_argument("--audit-args", default="", help="透传给 audit_tracks 的额外参数")
    ap.add_argument("--dump", action="store_true", help="同时打印完整审计文本")
    args = ap.parse_args()

    extra = args.audit_args.split() if args.audit_args else []
    rows = []
    for item in args.items:
        name, _, path = item.partition("=")
        if not path:
            name, path = os.path.basename(item), item
        if not os.path.exists(path):
            print(f"[skip] {name}: 找不到 {path}")
            continue
        text = run_audit(path, extra)
        m = parse(text)
        m["name"] = name
        rows.append(m)
        if args.dump:
            print(f"=== {name} ===")
            print(text)
            print()

    cols = ["name", "tracks", "emitted", "broken", "stitched",
            "missing_seg", "missing_max", "overlap_bad", "area_jump", "shape_burst"]
    heads = ["tracker", "轨迹数", "编号", "真断裂", "缝合",
             "缺失段", "最长缺失", "重叠超标", "面积跳变", "形状突变"]
    print("=" * 96)
    print("多跟踪器审计对比  (遮挡相关：真断裂/缺失段/最长缺失/面积跳变 越少越好)")
    print("=" * 96)
    print(" | ".join(f"{h:>8s}" for h in heads))
    print("-" * 96)
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:>8}" if v is not None else f"{'-':>8}")
        print(" | ".join(cells))
    print("=" * 96)


if __name__ == "__main__":
    main()
