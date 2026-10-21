# 多边形维诺图：64 点轮廓与重叠连续折线

本目录整理了 `14-10.20.mp4` 全片处理所用算法，提供可配置的运行入口、中文说明、原始脚本快照和验证记录。当前主流程使用 CPU、Numba 和多进程，不需要 GPU 或重新运行 YOLO。

核心约定：每个掩码保留 **64 个轮廓点及其组成的 64 条线段**；两个掩码重叠时，将全部有序交点连成 **1–2–3–4…的开放连续折线**。普通维诺网络只显示在掩码外，掩码内另外绘制约定的重叠分界。

## 目录

| 文件或目录 | 用途 |
|---|---|
| `render_video.py` | 视频处理入口，支持配置文件、帧范围、预览和多进程 |
| `bvh_field.py` | 线段 BVH，查询每个网格点最近的轮廓线段 |
| `topology_voronoi.py` | 占据判定、完整轮廓求根、网络连接、掩码裁剪 |
| `overlap_chords.py` | 交点排序、连续折线、重叠区域划分；名称保留历史命名，实际支持多段折线 |
| `prepare_tracks.py` | 从原始 pass1 缓存重建与 14-10.20 视频一致的 64 点轮廓 |
| `tracking_postprocess.py` | 原跟踪脚本快照，预处理入口只调用其离线后处理函数 |
| `_deps.py` | 可选本地依赖目录支持 |
| `run_14.ps1` | 当前电脑的一键运行入口 |
| `configs/14-10.20.json` | 本例视频、缓存、输出路径和进程数 |
| `requirements.txt` | 本次验证使用的依赖版本 |
| [算法说明.md](算法说明.md) | 几何定义、处理流程、效率来源和异常处理 |
| [复现与验证.md](复现与验证.md) | 缓存格式、预处理参数、验证方法和实测结果 |
| `evidence/` | 原全片的统计、异常索引及预览图 |
| `archive/original_full14/` | 全片运行时的原始脚本，保留原路径假设，供追溯 |
| `check_package.py`、`tests/` | 整理后的代码检查及检查产物 |

## 当前电脑直接运行

在 PowerShell 中：

```powershell
Set-Location 'E:\AAA-reseach\packet\voronoi-new'

# 先输出 6 帧预览 PNG，不生成视频
.\run_14.ps1 -Preview

# 完整处理，默认 3 个帧处理进程
.\run_14.ps1

# 仅处理从第 1400 帧开始的 30 帧，另存结果
.\run_14.ps1 -Start 1400 -Count 30 -OutputDir '.\outputs\clip1400'
```

`run_14.ps1` 默认使用现有 `E:\AAA-reseach\env\yolo26\python.exe` 及实验目录中的依赖；它只为此次运行设置依赖路径，退出时恢复环境变量。默认输入路径在配置文件内，输出位于本目录的 `outputs/14-10.20/`。

已有同名 MP4 时，程序会停止；需要重新生成时传入 `-Overwrite`，或换输出目录。

## 换环境运行

使用 Python 3.11，安装依赖后即可直接运行。依赖文件记录的是本次实际使用的版本；改变版本后应重新检查数值结果。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe render_video.py `
  --source 'D:\data\video.mp4' `
  --cache 'D:\data\frames64.pkl' `
  --out-dir '.\outputs\new-video' `
  --workers 3
```

也可以使用 `--config configs/14-10.20.json`。配置内相对路径以 JSON 所在目录为基准；命令行相对路径以当前工作目录为基准。命令行参数优先于配置文件。

主入口接收已经对齐视频帧的轮廓缓存，不负责从任意新视频中检测掩码。缓存格式及本例缓存重建方法见 [复现与验证.md](复现与验证.md)。当前入口固定使用每个掩码 64 点；坐标为原图像素坐标，图像宽高要求为偶数。当前编码输出不含音轨；本例源视频本身没有音轨。

## 输出内容

- `*_voronoi64_polyline.mp4`：白色网络叠加视频，H.264、CRF 18、原图尺寸和原帧率。
- `frames.jsonl`：逐帧交点坐标、对象编号、计算耗时、修复和异常。
- `validation.json`：完整解码验证、计时统计、输入路径与脚本哈希。
- `result_f*.png`：抽查帧；预览模式对应 `preview_f*.png` 和 `preview_stats.json`。
- `encoder.log`：FFmpeg 日志。

原完整视频和约 113 MB 的轮廓缓存通过配置引用，未重复复制进代码包。本目录的代码不需要导入原实验目录中的 Python 模块；在新环境安装依赖并提供视频、缓存后可独立运行。

## 当前完成范围

原全片共 2901 帧，2510 × 676，30 fps，96.7 秒，已完整解码验证。渲染和编码耗时约 323.8 秒；共生成 14168 条“对象对 × 帧”的重叠折线。

本算法是网格数值重建，并非解析精确的线段维诺图。原视频有 89 帧涉及交点排序不一致、三重重叠或包含，这些位置未强行连接；外部网络仍可能因为求根或三岔点求解失败留下小缺口。整理代码没有宣称消除了这些问题，详细计数保留在 `evidence/`。
