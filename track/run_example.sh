#!/usr/bin/env bash
# ============================================================================
# 完整复现命令（按需修改顶部 5 个路径）
# ============================================================================
set -euo pipefail

# ---- 按你的环境改这 5 个 ----
PY="E:/AAA-reseach/env/yolo26/python.exe"          # 带 ultralytics 的 python
ROOT="E:/AAA-reseach"                              # 项目根（脚本与视频都在其下）
MODEL="$ROOT/round4_cropped_1536_r3_cp05/round4_cropped_1536_r3_cp05/weights/best.pt"
SRC="$ROOT/videos/new/14.mp4"
OUTDIR="$ROOT/videos/new-track"

NAME="14_track"          # 输出文件名（不带扩展名）
IMGSZ=1536               # ★ 必须与训练尺度一致
TRACKER="fasttrack.yaml" # 或 shape.yaml（需先按 ultralytics_patch/ 安装）

# ---- 一般不用改 ----
# --conf      跟踪门限：已建轨迹续帧用
# --conf-init 首现门限：新建轨迹用
# --no-crop   全尺寸输出（默认会按黑边裁上下）
"$PY" -u "$ROOT/tools/track_segment.py" \
  --source  "$SRC" \
  --out     "$OUTDIR/$NAME.mp4" \
  --out-dir "$OUTDIR" \
  --model   "$MODEL" \
  --tracker "$TRACKER" \
  --imgsz   "$IMGSZ" \
  --conf 0.3 \
  --conf-init 0.75 \
  --no-crop \
  2>&1 | tee "$OUTDIR/${NAME}.log"

# ---- 交付前必查：帧数必须等于源视频 ----
"$PY" - <<PY
import cv2, os, sys
src, out = r"$SRC", r"$OUTDIR/$NAME.mp4"
def info(p):
    c = cv2.VideoCapture(p)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); w = int(c.get(3)); h = int(c.get(4))
    c.release(); return n, w, h
n0 = info(src); n1 = info(out)
print(f"源 {n0[0]} 帧 {n0[1]}x{n0[2]} -> 输出 {n1[0]} 帧 {n1[1]}x{n1[2]}  "
      f"({os.path.getsize(out)/1024**2:.1f} MB)")
sys.exit(0 if n1[0] == n0[0] else f"!! 帧数不一致，编码可能被中断（moov atom missing）")
PY
