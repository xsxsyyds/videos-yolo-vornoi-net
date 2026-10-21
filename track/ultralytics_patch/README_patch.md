# 自研 7D 形状感知跟踪器 —— 安装说明

`trackers/shape_tracker.py` 不属于官方 ultralytics，需要手工注册后才能用
`--tracker shape.yaml` 调起。

> ⚠️ **默认流水线不使用它**。在 9.mp4 上它只与官方 fasttrack 打平、最长缺失更差；
> 在 12/14 上缝合段数明显更多；形状项对 vehicle 干扰大。详见 `../README.md` §5.1 与
> `../docs/7D形状感知跟踪器_实现与评测.md`。

## 安装三步

### 1. 放文件

```
<site-packages>/ultralytics/trackers/shape_tracker.py
```

### 2. 注册（两个文件）

`ultralytics/trackers/track.py`：

```python
from .shape_tracker import ShapeTracker          # 加在其它 tracker 的 import 旁边

TRACKER_MAP = {
    ...
    "shape": ShapeTracker,                       # 加到映射表里
}
```

`ultralytics/trackers/__init__.py`：

```python
from .shape_tracker import ShapeKalmanFilter7D, ShapeSTrack, ShapeTracker

__all__ = (..., "ShapeKalmanFilter7D", "ShapeSTrack", "ShapeTracker")
```

### 3. 放配置

```
<site-packages>/ultralytics/cfg/trackers/shape.yaml
```

（内容就用本包的 `../tools/trackers/shape.yaml`）

> **必须放到 `cfg/trackers/` 下**，否则 `--tracker shape.yaml` 会解析不到文件。

## 使用

```bash
python -u tools/track_segment.py \
  --source "<视频>.mp4" --out "<输出>.mp4" --out-dir "<输出目录>" \
  --model "<权重>/best.pt" --tracker shape.yaml --imgsz 1536 --no-crop

# ⚠️ 改了跟踪器代码务必删缓存，否则会"命中缓存"、结果不变：
rm -f "<输出目录>/<名字>_pass1.pkl"
```

## 两个致命坑（改这个文件前必读）

1. **必须覆盖 `STrack.tlwh` / `xywh`**
   基类假定 `mean[:4] = (cx, cy, w, h)`，而 7D 状态里 `mean[2:4]` 是**速度**。
   不覆盖 → bbox 退化成一条线 → IoU 恒为 0/1 → **跟踪完全失效**
   （实测：两帧只差 2px 的框，IoU 距离返回 1.0）。
   正确做法是由面积与宽高比反解：`h = sqrt(area/aspect)`、`w = aspect*h`。

2. **`lost_stracks` 必须按 `max_frames_lost` 淘汰**
   原实现只 append 从不清理 → 早期丢失的轨迹永远留在匹配池里，
   几百帧后被不相干目标重新匹配上（实测有条轨迹只有 30 个真帧，却横跨 0→712 帧）。
   但光加淘汰会更糟：`track_buffer` 默认 30 太激进（554 轨迹 / 206 断裂），
   **必须同时把 `track_buffer` 提到 90**。

## 论文阈值不能照搬

参考论文的消融说 IoU 阈值 **0.3 优于 0.5**，但那是**纯掩膜 IoU** 代价。
本实现的门限卡在 **IoU ⊕ 形状的融合值**上，同一个 0.3 等于一扇紧得多的门
（实测 89 轨迹 / 7 断裂）。调到 **0.70** 后变成 80 轨迹 / 0 断裂。

> 只有当代价函数的定义与量纲一致时，外部阈值才可直接借用。
