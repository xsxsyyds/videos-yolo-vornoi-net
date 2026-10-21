# 无人机俯拍交通视频 —— 完整处理流水线

本目录是把一段无人机俯拍视频，做成「**稳像裁切 → 分割跟踪 → 多边形维诺网络**」成品的完整代码与说明。

三段各自独立可跑，但**顺序不能换、数据必须同源**。本文只讲三段怎么串起来用；
各段的算法细节在子目录自己的文档里（见 §12）。

> 📮 **关于 `track/best.pt` 模型权重**：因超出 GitHub 单文件 100 MB 上限，本仓库未附带该权重文件。
> 如需获取，请联系 QQ 邮箱：**2639022401@qq.com**。

```
原始视频 (2560×800)
   │
   ├─① stabilized/      增稳 + 最大公共画幅裁切
   │      → 14_stabilized_cropped.mp4 (2510×676，无黑边，静止物体固定)
   │
   ├─② track/           YOLO 分割 → 多目标跟踪 → 时序滤波 → 渲染
   │      → 14-10.20.mp4（带掩膜与「编号 置信度」）+ _pass1.pkl + _tracks.json
   │
   └─③ voronoi-new/     以 64 点轮廓为站点，画改版维诺网络
          → 14-10.20_voronoi64_polyline.mp4（叠加白色网络）
```

---

## 0. 三条命令（TL;DR）

在 `E:\AAA-reseach\packet` 下，PowerShell：

```powershell
Set-Location 'E:\AAA-reseach\packet'
$PY = 'E:\AAA-reseach\env\yolo26\python.exe'

# ① 增稳 + 裁切（产出 .\14_stabilized\14_stabilized_cropped.mp4）
& $PY .\stabilized\stabilize_static.py .\14.mp4 --output .\14_stabilized --stage all --model homography
& $PY .\stabilized\crop_stabilized.py  .\14.mp4 --output .\14_stabilized

# ② 分割 + 跟踪（产出 .\outputs\14-10.20.mp4）
& $PY .\track\tools\track_segment.py `
    --source .\14_stabilized\14_stabilized_cropped.mp4 `
    --out    .\outputs\14-10.20.mp4 `
    --model  .\track\best.pt `
    --tracker fasttrack.yaml --imgsz 1536 --conf 0.3 --conf-init 0.7 --no-crop

# ③ 维诺网络（先建 64 点轮廓缓存，再出片）
& $PY .\voronoi-new\prepare_tracks.py `
    --pass1 .\outputs\14-10.20_pass1.pkl `
    --tracks-json .\outputs\14-10.20_tracks.json `
    --out-dir .\voronoi-new\outputs\prepared

& $PY .\voronoi-new\render_video.py `
    --source  .\outputs\14-10.20.mp4 `
    --cache   .\voronoi-new\outputs\prepared\frames64.pkl `
    --out-dir .\voronoi-new\outputs\14-10.20
```

> 路径都可用自己的。**唯一的硬约束**是 §6 那五条「衔接契约」——尤其是
> ③ 的 `--source` 必须是 ② 的成片，`--cache` 必须由 ② 的 `_pass1.pkl` 生成。
>
> ⚠️ `voronoi-new\run_14.ps1` 只是把 `configs\14-10.20.json` **原样**喂给渲染器，
> 而那份配置里写的是**原实验的绝对路径**（`E:\AAA-reseach\videos\new-track\...`）。
> 所以：要么按上面的写法显式传 `--source` / `--cache` / `--out-dir`，
> 要么先把 `configs\14-10.20.json` 里这三项改成你自己的路径。

---

## 1. 目录总览

| 目录 | 入口 | 做什么 | 需要什么 |
|---|---|---|---|
| `stabilized/` | `stabilize_static.py`、`crop_stabilized.py`、`review_stabilized.py` | 相机运动补偿 + 固定画幅裁切 + 独立验收 | Python + OpenCV（CPU，数分钟） |
| `track/` | `tools/track_segment.py`（`best.pt` 权重不在仓库内，见文首声明） | 分割、跟踪、去重、缝合、伪影清理、平滑、渲染 | **GPU**（RTX 5060 约 8~10 fps）+ ultralytics（见 §9） |
| `voronoi-new/` | `prepare_tracks.py` + `render_video.py`（或 `run_14.ps1`） | BVH 最近线段场 + 拓扑网络 + 重叠折线 + 编码 | **CPU**，需 numba / shapely，**不跑 YOLO、不需要 GPU** |

---

## 2. 阶段一：stabilized —— 增稳与裁切

### 2.1 为什么放第一步

无人机悬停有风扰，静止地面物体在画面里会漂 6 像素左右（实测中位数 6.177 px，95% 分位 12.401 px）。
如果先做分割，跟踪器要额外对付「整幅画面平移」；而稳像之后再分割，所有下游坐标都在**同一个稳定坐标系**里，
跟踪的 IoU 匹配、缝合外推、维诺建场都更稳。

裁切不是"为了好看"：稳像把每帧重采样到固定画布后，相机运动推出去的部分会变成黑边；
裁切按**全片交集**求出最大公共矩形，一次去掉黑边并锁定画幅（本例 2560×800 → **2510×676**，面积减少约 17%），
这样后面的分割与维诺都不必逐帧处理"画面边界在哪"。

### 2.2 命令

> 下面假设源视频 `14.mp4` 就在包根目录；不在的话把 `.\14.mp4` 换成绝对路径即可。

```powershell
$PY = 'E:\AAA-reseach\env\yolo26\python.exe'
Set-Location 'E:\AAA-reseach\packet'

# 一次性跑完：搭背景 → 估计变换 → 导出稳像视频
& $PY .\stabilized\stabilize_static.py .\14.mp4 --output .\14_stabilized --stage all --model homography

# 最大公共画幅裁切（读取上一步的 transforms_homography.npz）
& $PY .\stabilized\crop_stabilized.py .\14.mp4 --output .\14_stabilized

# 独立验收（模板匹配 + ECC 亚像素，不复用拟合点）
& $PY .\stabilized\review_stabilized.py .\14.mp4 --output .\14_stabilized
```

也可分步跑：`--stage prepare / estimate / render`。

**模型选择**：`--model similarity / affine / homography`。本例抽样对比（49 帧、同一检查方法）：
similarity 残差中位 0.162 px、affine 0.159 px、**homography 0.113 px**（95% 分位 0.486 px），
故默认用 `homography`。

### 2.3 产物

| 文件 | 内容 |
|---|---|
| `<名>_stabilized.mp4` | 稳像后完整视频（原分辨率、帧率、帧数，边缘填黑，保留音轨） |
| **`<名>_stabilized_cropped.mp4`** | **给阶段二用的输入**（固定裁切框，无黑边） |
| `transforms_homography.npz` | 每帧 3×3 变换 + 检查点误差；方向为「参考坐标 → 原始帧坐标」 |
| `transforms_homography_cropped.npz` | 裁切坐标 → 原片坐标的每帧变换 |
| `crop_geometry.json` | 裁切框坐标、尺寸、保留面积 |
| `crop_common_valid_mask.png` | 所有帧有效区域的交集 |
| `14_center_comparison.mp4` | 开头/中间/结尾各 5 秒左右并排对比（蓝标固定在参考坐标） |
| `reference_points.jpg`、`residuals.png`、`error_distribution.png` | 参考点分布与逐帧误差 |
| `处理报告.md` | 结果、误差统计、异常帧、模型比较 |

### 2.4 三个必须知道的点

1. **变换矩阵方向是「参考 → 原帧」**，导出时用 `WARP_INVERSE_MAP`。
   若要把**原片里的检测坐标**转到稳定坐标，要用**该矩阵的逆矩阵**。
2. **裁切只做固定偏移**：`x_new = x_stabilized − crop_x`，`y_new = y_stabilized − crop_y`，
   全片同一个框，所以静止物体不会因为裁切而逐帧移动。
3. **不改原片、不覆盖成品**。成品已存在时导出会停止（防止误覆盖上一次的结果）；
   要重出就换输出目录，或先移走旧成品。每段视频必须重建自己的参考背景，
   **不能复用 14.mp4 的变换矩阵**。

### 2.5 实测（14.mp4）

| 项 | 值 |
|---|---|
| 原片检查点位移 中位 / 95% | 6.177 / 12.401 px |
| 成品残余误差 中位 / 95% | **0.099 / 0.372 px** |
| 有效观测误差 < 0.5 px / < 1 px | 97.53% / 99.64% |
| 参与拟合 / 独立检查点 | 1459 / 46 |
| 裁切后尺寸 | 2510 × 676（偶数，适配 yuv420p） |

---

## 3. 阶段二：track —— 识别、分割、跟踪

### 3.1 做什么

一次运行完成五层处理，输出带掩膜标注的成片，同时把逐帧实例缓存下来给阶段三用。

| 层 | 内容 | 关键参数（默认） |
|---|---|---|
| ① 跟踪器 | Ultralytics **FastTracker**（ByteTrack 遮挡感知变体）：卡尔曼 XYAH + 两阶段关联 + 遮挡回滚 | `track_high_thresh 0.25`、`track_buffer 30`、`occ_cover_thresh 0.7`、`init_iou_suppress 0.7` |
| ② 两级置信度 | 检测跑 `conf`（续帧门槛低），**建新轨迹**要求 `conf-init`（门槛高） | `--conf 0.3`、`--conf-init 0.7` |
| ③ 后处理 | 身份感知去重 → 轨迹缝合 → 粘连掩膜修正 → 重复掩膜修正 → 短轨迹过滤 | `dedup 0.5/0.8/0.9`、`stitch_gap 40`、`merge_max_len 12`、`overlap_jump 0.80` |
| ④ 时序平滑 | 插值 + Savitzky-Golay + 轮廓对齐/限速/收边 + **缺帧填补** | `smooth_win 15`、`poly_win 11`、`poly_fill_gap 20` |
| ⑤ 渲染编码 | 掩膜套色 + `编号 置信度` 标签，管道直送 ffmpeg | `--crf 20`、不画矩形框 |

### 3.2 命令

```powershell
& $PY .\track\tools\track_segment.py `
    --source .\14_stabilized\14_stabilized_cropped.mp4 `
    --out    .\outputs\14-10.20.mp4 `
    --model  .\track\best.pt `
    --tracker fasttrack.yaml `
    --imgsz 1536 --conf 0.3 --conf-init 0.7 --no-crop
```

**四个必须显式给的参数**：

| 参数 | 为什么必须给 |
|---|---|
| `--model` | 默认值指向旧权重 `runs/segment/round3_744`，不是本包附带的 `best.pt` |
| `--imgsz 1536` | 模型训练于 1536，**推理尺度必须与训练一致**（默认 1280 是错的） |
| `--no-crop` | 输入已经裁过黑边，再裁会改变输出尺寸，下游缓存就要跟着重算 |
| `--out` | **pass1 缓存与 tracks.json 都跟随 `--out`**（`<out>_pass1.pkl` / `<out>_tracks.json`），不是跟随 `--out-dir` |

> `--tracker fasttrack.yaml` 由 ultralytics 从**它自己的** `cfg/trackers/` 解析，
> 用的是 `track_buffer: 30`。包内 `track/tools/trackers/` 是若干实验副本
> （`official_fasttrack.yaml` 把 `track_buffer` 提到 90、`shape.yaml` 是自研 7D 跟踪器），
> **要生效必须传完整路径**，例如 `--tracker .\track\tools\trackers\official_fasttrack.yaml`。

### 3.3 产物

| 文件 | 内容 |
|---|---|
| `14-10.20.mp4` | 带掩膜与标签的成片（阶段三的输入视频） |
| `14-10.20_pass1.pkl` | 逐帧跟踪缓存（阶段三建轮廓缓存的原料） |
| `14-10.20_tracks.json` | 轨迹清单（编号、类别、首末帧、平均置信度） |

### 3.4 实测（14_stabilized_cropped + 首现 0.7）

2901 帧、2510×676；pass1 341 s（8.5 fps，GPU）；轨迹 80 条（车 2 / 骑手 41 / 行人 37）；
每帧目标平均 72.6、最多 74；缝合 10 段、粘连修正 3 处、重复掩膜 7 帧、缺帧填补 189 帧；成片 68.6 MB。

---

## 4. 阶段三：voronoi-new —— 多边形维诺网络

### 4.1 与旧版的关键区别

| | 旧版（`archive/original_full14`） | **voronoi-new** |
|---|---|---|
| 站点 | 轮廓抽稀到 6 点 | **保留全部 64 点、64 条线段** |
| 距离计算 | 每对象一整张稠密距离场 | **线段 BVH 最近邻查询**（带空间剪枝） |
| 掩码内部 | 用「到边界距离」参与分区 → 会出现内部小环 | **单独做占据判定**，内部不参与外部争抢 |
| 求根 | 固定两个候选线段 | **扫描该掩码全部线段**，允许最近线段在求根中切换 |
| 连接 | 按交点距离随意连线 | **三角网格连接 + 三岔点 Newton 求解** |
| 重叠 | 无统一约定 | **全部有序交点连成 1–2–3–4… 连续开放折线** |

### 4.2 为什么是两步

`render_video.py` 只接受**已经与视频逐帧对齐**的 64 点轮廓缓存（`frames64.pkl`），
它自己不跑检测。缓存必须由阶段二的 `_pass1.pkl` 重建——因为直接拿 pass1 的原始轮廓
会与成片里画的掩膜不一致（成片经过了缝合、伪影清理、平滑、缺帧填补）。

### 4.3 命令

```powershell
# ① 由 pass1 重建 64 点轮廓缓存（内部有断言：轨迹 ID/类别/编号/首末帧必须与 json 一致）
& $PY .\voronoi-new\prepare_tracks.py `
    --pass1       .\outputs\14-10.20_pass1.pkl `
    --tracks-json .\outputs\14-10.20_tracks.json `
    --out-dir     .\voronoi-new\outputs\prepared

# ② 出片（推荐：显式传三个路径，不受配置里的原始实验路径影响）
& $PY .\voronoi-new\render_video.py `
    --source  .\outputs\14-10.20.mp4 `
    --cache   .\voronoi-new\outputs\prepared\frames64.pkl `
    --out-dir .\voronoi-new\outputs\14-10.20 `
    --workers 3

# 先看 6 张预览 PNG 再出全片（同样是显式传参）
& $PY .\voronoi-new\render_video.py --preview `
    --source .\outputs\14-10.20.mp4 `
    --cache  .\voronoi-new\outputs\prepared\frames64.pkl `
    --out-dir .\voronoi-new\outputs\preview
```

也可用一键脚本（**前提**：先把 `configs\14-10.20.json` 的 `source` / `cache` / `out_dir`
改成自己的路径，否则它会去读原实验的绝对路径）：

```powershell
Set-Location .\voronoi-new
.\run_14.ps1 -Preview                     # 6 张预览 PNG，不生成视频
.\run_14.ps1                              # 全片，默认 3 个进程
.\run_14.ps1 -Start 1400 -Count 30 -OutputDir '.\outputs\clip1400'
```

配置文件 `configs/14-10.20.json` 里写死了本例的路径（源视频、缓存、输出目录、文件名、进程数）；
命令行参数优先于配置文件，相对路径以**当前工作目录**为基准（配置里则以 JSON 所在目录为基准）。

### 4.4 产物

| 文件 | 内容 |
|---|---|
| `*_voronoi64_polyline.mp4` | 白色网络叠加视频，H.264 / CRF 18 / 原尺寸原帧率 |
| `frames.jsonl` | 逐帧交点坐标、对象编号、耗时、修复与异常 |
| `validation.json` | 完整解码验证、计时、输入路径与脚本哈希 |
| `result_f*.png` / `preview_f*.png` | 抽查帧 |
| `encoder.log` | FFmpeg 日志 |

### 4.5 实测（14-10.20 全片）

| 项 | 值 |
|---|---|
| 帧数 / 尺寸 / 时长 | 2901 帧、2510×676、96.7 s |
| 网格 | `scale=0.5`（每 2 个原图像素一个查询点；**减少的是查询点数，不减少 64 点轮廓**） |
| 轮廓预处理 | 149.36 s |
| 渲染 + 编码 | 323.76 s（约 **8.96 帧/秒**） |
| 单帧平均耗时 | 321.52 ms（进程内计时，不能直接取倒数当多进程吞吐） |
| 输出体积 | 144,696,570 字节 |
| 重叠折线 | **14168 条**（2 点 12796 / 4 点 1319 / 6 点 47 / 8 点 6） |

---

## 5. 三段的数据流（记这张表就够）

| 段 | 输入 | 输出 | 下一段用哪个 |
|---|---|---|---|
| ① | 原始 `<名>.mp4` | `<名>_stabilized.mp4`、**`<名>_stabilized_cropped.mp4`** | ② 的 `--source` 用 **cropped** 那个 |
| ② | `_stabilized_cropped.mp4` | `X.mp4`、`X_pass1.pkl`、`X_tracks.json` | ③ 的 `--source` 用 `X.mp4`，`prepare_tracks` 用 `X_pass1.pkl` + `X_tracks.json` |
| ③ | `X.mp4` + `frames64.pkl` | `X_voronoi64_polyline.mp4` | — |

**一句话**：② 的输出视频既是交付物，也是 ③ 的画布；③ 的几何必须来自**同一份** `_pass1.pkl`。

---

## 6. ★ 衔接契约（这五条踩了就白跑）

1. **不要拿原始 `14.mp4` 的坐标去套裁切后的视频。**
   稳像 + 裁切把坐标系换了两次；要把原片坐标搬到稳定裁切坐标，得用
   `transforms_homography.npz` 的**逆矩阵**，再做裁切偏移。
   正常流程下不需要这个转换——因为 ② 直接在裁切后的视频上跑检测。

2. **`--cache` 必须由 ② 的 `_pass1.pkl` 生成，且与 ③ 的 `--source` 出自同一次运行。**
   帧数、宽高、帧率任何一项对不上，`render_video.py` 会直接报错；对上了但来源不同，
   线会整体错位。

3. **`prepare_tracks.py` 里的后处理参数是硬编码的，必须与 ② 实际用的参数一致。**
   它复刻的正是 ② 的离线后处理链路：

   | 步骤 | 硬编码参数 |
   |---|---|
   | 轨迹拼接 | `max_gap=40, max_res=90, 边界 60px`（`max_angle=60` 仅为兼容保留，方向判断已移除） |
   | 粘连掩码清理 | `area_ratio=1.45, cover=0.5, look=3, max_len=12` |
   | 重叠冲突清理 | `jump=0.8, base_thr=0.25, drop_conf=0.6, center_ratio=0.35` |
   | 短轨迹过滤 | 少于 5 帧丢弃 |
   | 框平滑 / 间隙 | `win=15, max_gap=45` |
   | 轮廓平滑 | `poly_win=11, poly_outlier=2.2` |
   | 形状限速 / 收边 | `shape_max=0.06, clamp_tol=0.03` |
   | 掩码缺口插值 | `poly_fill_gap=20, fill_max_body=2.5` |

   如果你改了 ② 的 `--stitch-gap`、`--merge-*`、`--overlap-*`、`--poly-fill-gap` 等，
   **必须同步改 `prepare_tracks.py`**，否则缓存里的轮廓和成片里画的对不上（断言可能直接失败）。

4. **不要在 ② 用 `--crop`。** 输入已经没有黑边，再裁会改变输出高度；
   虽然 ③ 的缓存也跟着同一次 pass1 走、仍然自洽，但你的成片尺寸会和预期不一致。

5. **不要混用不同视频的 `frames64.pkl`。** 缓存里没有存视频哈希，只有宽高帧数；
   两段同尺寸视频互换会静默错位。

---

## 7. 参数联动：改一个会牵连哪些

| 改动 | 直接影响 | 必须跟着做 |
|---|---|---|
| ① 换 `--model`（similarity/affine/homography） | 稳像质量、黑边范围 | 重跑裁切 → **② ③ 全部重跑** |
| ① 换 `--output` 目录 | 产物位置 | ② 的 `--source` 路径要跟着改 |
| ② 改 `--conf` / `--conf-init` | 轨迹数、编号 | **③ 的缓存必须重建**（轨迹变了） |
| ② 改 `--imgsz` | 检测尺度（必须 = 1536） | 同上 |
| ② 改缝合/清理/平滑参数 | 掩膜形状 | **同步改 `prepare_tracks.py`** 后重建缓存 |
| ② 改 `--out` 名字 | 三个产物文件名一起变 | ③ 的两条路径都要改 |
| ③ 改 `--workers` | 只影响速度 | 无（结果与进程数无关） |
| ③ 改网格 `scale` | 网络精度/速度 | 属于算法改动，需重新验证 |

---

## 8. 一次完整跑通（以 14.mp4 为例）

```powershell
$PY = 'E:\AAA-reseach\env\yolo26\python.exe'
Set-Location 'E:\AAA-reseach\packet'
New-Item -ItemType Directory -Force .\outputs | Out-Null

# ---------- ① 增稳 + 裁切（耗时随帧数增长，本例 2901 帧为分钟级到十几分钟） ----------
& $PY .\stabilized\stabilize_static.py .\14.mp4 --output .\14_stabilized --stage all --model homography
& $PY .\stabilized\crop_stabilized.py  .\14.mp4 --output .\14_stabilized
# 目检：.\14_stabilized\14_center_comparison.mp4 与 .\14_stabilized\处理报告.md

# ---------- ② 分割 + 跟踪（2901 帧约 10 分钟，GPU） ----------
& $PY .\track\tools\track_segment.py `
    --source .\14_stabilized\14_stabilized_cropped.mp4 `
    --out    .\outputs\14-10.20.mp4 `
    --model  .\track\best.pt `
    --tracker fasttrack.yaml `
    --imgsz 1536 --conf 0.3 --conf-init 0.7 --no-crop
# 交付前校验：输出帧数必须等于输入帧数

# ---------- ③ 维诺网络 ----------
& $PY .\voronoi-new\prepare_tracks.py `
    --pass1       .\outputs\14-10.20_pass1.pkl `
    --tracks-json .\outputs\14-10.20_tracks.json `
    --out-dir     .\voronoi-new\outputs\prepared

# 先预览 6 帧，确认线位置正确再出全片
& $PY .\voronoi-new\render_video.py --preview `
    --source  .\outputs\14-10.20.mp4 `
    --cache   .\voronoi-new\outputs\prepared\frames64.pkl `
    --out-dir .\voronoi-new\outputs\preview

& $PY .\voronoi-new\render_video.py `
    --source  .\outputs\14-10.20.mp4 `
    --cache   .\voronoi-new\outputs\prepared\frames64.pkl `
    --out-dir .\voronoi-new\outputs\14-10.20 `
    --workers 3
```

> 本示例从包根目录执行，因此 ① 的输入写的是 `.\14.mp4`——把你的源视频放到包根目录，
> 或者把这一处改成绝对路径。

---

## 9. 环境与依赖

| 段 | 解释器 | 关键依赖 |
|---|---|---|
| ① | 任意 Python 3.10+ | `numpy≥2.0`、`opencv-python-headless≥4.8`、`scipy≥1.11`、`matplotlib≥3.8`、`imageio-ffmpeg≥0.6`（见 `stabilized/requirements-stabilize.txt`） |
| ② | **`E:\AAA-reseach\env\yolo26\python.exe`**（3.11.16） | **本机 ultralytics（yolo26 分支，8.4.154）**、torch 2.11+cu128、opencv 5.0.0、ffmpeg |
| ③ | 同 ② 或独立 3.11 venv | `numpy 2.4.6`、`opencv 5.0.0.93`、`numba 0.68.0`、`shapely 2.2.0`、`imageio-ffmpeg` |

两个**环境坑**：

- **② 的 `fasttrack.yaml` 来自本机 ultralytics 检出目录**（`E:\AAA-reseach\yolo26\ultralytics\cfg\trackers\`），
  不是 PyPI 版就一定有。换环境前先确认该文件存在，否则用 `track/tools/trackers/official_fasttrack.yaml` 并显式传完整路径。
- **ffmpeg 探测**：不能只信 `shutil.which('ffmpeg')`——环境里可能有同名坏 shim，
  `which` 认为存在但执行报 `WinError 2`。正确做法是对候选真跑一次 `-version`。

---

## 10. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| ② 报 `source not found` | ① 的产物名不是 `*_stabilized_cropped.mp4`，或路径写错 | 到 ① 的输出目录确认实际文件名（默认是 `<源名>_stabilized_cropped.mp4`） |
| ② 报找不到 `fasttrack.yaml` | ultralytics 不是本机那个分支 | 见 §9，改用 `official_fasttrack.yaml` 并传完整路径 |
| ② 输出帧数少于输入 | 编码被中断（`moov atom not found`） | 用 `cv2.CAP_PROP_FRAME_COUNT` 核对，重跑；不要只看文件大小 |
| ② 跑得很慢（<1 fps） | 掉到 CPU 了 | 看启动头部 `device : cuda:0 — …` |
| ③ 报「Output exists」 | 默认不覆盖 | `render_video.py` 加 `--overwrite` 或换 `--out-dir`；`run_14.ps1` 用 `-Overwrite` / `-OutputDir` |
| ③ 线整体错位 | `--cache` 与 `--source` 不同源 | 用同一次 ② 的产物重建缓存 |
| ③ `prepare_tracks` 断言失败 | ② 的后处理参数与硬编码不一致 | 见 §6 第 3 条，同步参数 |
| ③ 报缺 shapely / numba | 依赖没装 | `pip install -r voronoi-new/requirements.txt` |

---

## 11. 已知限制

**稳像**：单个单应性主要稳定**同一地面平面**；悬空隔离带、灯杆、行人车辆的自身运动、
不同高度产生的视差不会被消除。误差数字是图像匹配估计，不是外部真值。

**跟踪**：粘连掩膜与重复掩膜的处理是启发式，极端拥挤处仍可能漏人或误删；
缺帧填补是插值，不是重新检测。

**维诺**：本算法是**网格数值重建，不是解析精确的线段维诺图**。原视频有 89 帧涉及
交点排序不一致（69）、三重重叠（24）、包含或不足两个交点（1），这些位置**未强行连接**；
外部网络仍可能因求根或三岔点失败留下小缺口。当前也**没有时间维度的拓扑稳定性约束**，
轮廓平滑不能保证网络完全不闪。

---

## 12. 子文档索引

| 文档 | 内容 |
|---|---|
| `stabilized/STABILIZATION_README.md` | 稳像算法、分步命令、验收方法、输出文件清单 |
| `stabilized/裁切说明.md`、`crop_note.md` | 最大公共画幅裁切的定义与几何 |
| `stabilized/14处理报告.md` | 14.mp4 的实测误差、异常帧统计、模型比较 |
| `track/README.md` | 跟踪流水线全链路（五层）、参数总表、关键设计决策与踩坑 |
| `track/docs/7D形状感知跟踪器_实现与评测.md` | 自研跟踪器的实现与对比（默认不启用） |
| `voronoi-new/算法说明.md` | 几何定义、处理流程、BVH 剪枝、重叠折线规则、异常含义 |
| `voronoi-new/复现与验证.md` | 缓存格式、`prepare_tracks` 参数、验证覆盖与实测结果 |
| `voronoi-new/archive/README.md` | 原始全片脚本快照（仅供追溯，不是运行入口） |
