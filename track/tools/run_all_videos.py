# -*- coding: utf-8 -*-
"""批量处理 1~8.mp4：分段跟踪 + 编号 + 平滑 + 渲染，并汇总审计报告

特点
----
* 每个视频完全独立（编号互不影响），符合「不同视频相互独立」的要求。
* 断点续跑：已完成的视频自动跳过（除非 --force）。
* 每段结束自动跑一次审计，结果汇总到 _summary.txt。

用法
----
    python tools/run_all_videos.py                  # 全部
    python tools/run_all_videos.py --videos 2 6     # 只跑指定视频
    python tools/run_all_videos.py --force           # 重跑已完成的
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, "env", "yolo26", "python.exe")
SCRIPT = os.path.join(HERE, "track_segment.py")
AUDIT = os.path.join(HERE, "audit_tracks.py")


def human(sec):
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, nargs="*", default=list(range(1, 9)))
    ap.add_argument("--videos-dir", default=os.path.join(ROOT, "videos"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "videos", "output", "track"))
    ap.add_argument("--model", default=os.path.join(
        ROOT, "runs", "segment", "round3_744", "weights", "best.pt"))
    # 默认改用官方 FastTracker：官方文档的场景决策里，「实时频繁局部重叠 +
    # 无 ReID 预算」对应 FastTracker（遮挡感知的 ByteTrack 变体，带 Kalman 回滚
    # 与 init-IoU 抑制）。原先用的 deepocsort 属 OC-SORT 系，其关联代价含 OCM
    # 速度方向一致性项，对经常变向的目标不友好。
    ap.add_argument("--tracker", default="fasttrack.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.6,
                    help="跟踪门限：检测置信度下限")
    ap.add_argument("--conf-init", type=float, default=0.8,
                    help="首现门限：新对象建立轨迹的置信度下限")
    ap.add_argument("--order", default="diag")
    ap.add_argument("--smooth-win", type=int, default=15)
    ap.add_argument("--poly-win", type=int, default=11)
    ap.add_argument("--shape-max", type=float, default=0.06)
    ap.add_argument("--clamp-tol", type=float, default=0.03)
    ap.add_argument("--poly-outlier", type=float, default=2.2)
    ap.add_argument("--stitch-gap", type=int, default=40)
    ap.add_argument("--font-scale", type=float, default=0.45)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-crop", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    t_all = time.time()

    n_done = 0
    for v in args.videos:
        src = os.path.join(args.videos_dir, f"{v}.mp4")
        out = os.path.join(args.out_dir, f"{v}_track.mp4")
        cache = os.path.join(args.out_dir, f"{v}_track_pass1.pkl")
        log = os.path.join(args.out_dir, f"{v}_track.log")
        if not os.path.isfile(src):
            print(f"[{v}] 跳过：{src} 不存在", flush=True)
            continue
        if os.path.isfile(out) and os.path.getsize(out) > 0 and not args.force:
            print(f"[{v}] 已完成，跳过（--force 可强制重跑）", flush=True)
            n_done += 1
            continue

        cmd = [PY, SCRIPT, "--source", src, "--out", out,
               "--model", args.model, "--tracker", args.tracker,
               "--imgsz", str(args.imgsz), "--conf", str(args.conf),
               "--conf-init", str(args.conf_init),
               "--order", args.order, "--smooth-win", str(args.smooth_win),
               "--poly-win", str(args.poly_win),
               "--shape-max", str(args.shape_max),
               "--clamp-tol", str(args.clamp_tol),
               "--poly-outlier", str(args.poly_outlier),
               "--stitch-gap", str(args.stitch_gap),
               "--font-scale", str(args.font_scale), "--crf", str(args.crf)]
        if args.no_crop:
            cmd.append("--no-crop")

        print(f"\n{'='*74}\n[{v}] {os.path.basename(src)} -> {os.path.basename(out)}",
              flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as lf:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        el = time.time() - t0
        ok = p.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0
        print(f"[{v}] {'完成' if ok else '失败'}  用时 {human(el)}"
              + (f"  体积 {os.path.getsize(out)/1024**2:.0f}MB" if ok else
                 f"  见日志 {log}"), flush=True)

        # 审计
        if ok and os.path.isfile(cache):
            r = subprocess.run([PY, AUDIT, cache], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            at = os.path.join(args.out_dir, f"{v}_audit.txt")
            with open(at, "w", encoding="utf-8") as f:
                f.write(r.stdout or "")
            if r.returncode == 0:
                print(f"[{v}] 审计 -> {os.path.basename(at)}", flush=True)
            summary.append((v, human(el), out))

    print(f"\n{'='*74}")
    print(f"总计用时 {human(time.time()-t_all)} | 完成 {len(summary)} 个"
          + (f" | 此前已完成 {n_done} 个" if n_done else ""))
    if summary:
        print("输出:")
        for v, el, o in summary:
            print(f"  {v}_track.mp4   {el}   {os.path.getsize(o)/1024**2:.0f}MB")
    print("=" * 74)


if __name__ == "__main__":
    main()
