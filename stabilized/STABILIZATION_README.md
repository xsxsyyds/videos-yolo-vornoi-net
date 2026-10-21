# 固定参考坐标的无人机视频稳像

目标：让画面中心附近的静止地面物体尽可能保持固定像素位置。
原始 MP4 不会被覆盖。每段视频使用自己的参考背景和固定坐标系。

## 使用方法

```powershell
python -m pip install -r requirements-stabilize.txt
python stabilize_static.py 14.mp4 --output 14_stabilized --stage prepare
python stabilize_static.py 14.mp4 --output 14_stabilized --stage estimate --model homography
python stabilize_static.py 14.mp4 --output 14_stabilized --stage render --model homography
python review_stabilized.py 14.mp4 --output 14_stabilized
```

也可使用 `--stage all` 顺序完成背景构建、估计和导出。成品已存在时导出会停止，避免覆盖；
需要重新导出时请指定新的输出目录并重新准备、估计。

先比较模型可用：

```powershell
python stabilize_static.py 14.mp4 --stage estimate --stride 60 --model similarity
python stabilize_static.py 14.mp4 --stage estimate --stride 60 --model affine
python stabilize_static.py 14.mp4 --stage estimate --stride 60 --model homography
```

对另一段视频，将输入名替换为对应文件名即可。建议逐段检查参考点分布和误差，
不同视频的遮挡、道路平面、镜头运动情况不同，不能直接复制 14.mp4 的变换矩阵。

## 实际算法

1. 全视频均匀抽取 41 帧，用 SIFT 特征和 MAGSAC 几何共识向首帧坐标粗对齐。
2. 对齐后计算时间中位背景及像素变化幅度，再向背景做一次粗配准并重建背景。
   仅从长期稳定区域选取分布均匀的参考点。背景只用于估计相机运动，绝不替代视频画面。
3. 每帧直接从固定背景向当前帧做金字塔 LK 亚像素跟踪。前一帧的变换仅提供初始位置，
   不逐帧累乘变换。前后向跟踪不一致、外观差异大的点，以及不符合整体几何的点会被剔除。
4. 稳健估计平移/旋转/缩放、仿射或单应性变换；使用中心加权的稳健最小二乘精修。
   优先保持中心附近地面稳定，同时利用周围参考点约束角度与透视。
5. 从原始彩色帧一次性重采样导出。保持原分辨率、帧率、帧数和固定画布；
   保留原音轨，不做动态裁剪、动态缩放、补画或背景替换。边缘使用黑色填充。

## 独立验收

- 从中心 50% 宽度、纵向 15%～90% 的区域选择独立检查点，并排除其周围的拟合点。
- 采用局部归一化相关模板匹配加平移 ECC 亚像素精修，测量检查点的位置。
  此检查不参与相机变换估计，也不复用拟合点的 LK 流。
- 检查窗口内也使用时间稳定性遮罩，避免窗口边缘的摆动隔离带影响位置测量；
  遮罩后的静态纹理不足时，整个窗口标为不可测。
- 检查窗口在预测位置周围 ±8 像素搜索。低相关、匹配不唯一、峰值到达搜索边界的点
  标为不可测；这类点不能视为零误差。不会按最终残余误差大小删除有效观测。
- 报告包含可测覆盖率、中位数、95% 分位数、最大误差和逐帧可测点数。
  超过精度目标或可测点不足的帧保留在输出中并在 CSV 标记，需要结合对比画面查看。
- 对导出的 MP4 再完整解码并测量一次，因此最终报告覆盖实际编码输出。
- `python test_stabilize_static.py` 使用已知亚像素位移、旋转、透视变换及人工遮挡，
  检查变换方向、恢复精度和独立测量方法。此测试需要先生成 14_stabilized/reference.npz。

这些数值是基于图像匹配的误差估计，并非外部测量真值；无法保证被遮挡区域和所有像素
在全部帧都严格零误差。一个单应性主要稳定同一地面平面，悬空隔离带、灯杆、行人、车辆
的自身运动或不同高度产生的视差不会被当作相机运动全部消除。

## 输出文件

- `14_stabilized.mp4`：完整稳像视频。
- `14_center_comparison.mp4`：开头、中间、结尾各 5 秒，左右并排，中心裁切保持 1:1 像素比例。
  蓝色标记固定在参考坐标，便于目测静物是否移动；标记仅存在于对比视频。
- `reference_points.jpg`：绿色为拟合点，橙色为独立检查点。
- `residuals.png` / `error_distribution.png`：成品的逐帧误差与分布。
- `metrics_homography.csv` / `summary_homography.json`：变换估计阶段的误差和异常标记。
- `encoded_metrics.csv` / `encoded_verification.json`：完整解码后的实际成品检查。
- `处理报告.md`：结果、误差统计、异常帧数量、精度适用范围和模型比较。
- `transforms_homography.npz`：每帧 3×3 变换、检查点误差和有效性。
  矩阵方向为“参考坐标 → 原始帧坐标”，导出时使用 `WARP_INVERSE_MAP`。
  若要把原视频中的检测坐标转换到稳定坐标，应使用该矩阵的逆矩阵。

统计中所有像素单位均对应原始 2560×800 分辨率，而非缩小的预览画面。

## 最大公共画幅固定裁切

```powershell
python crop_stabilized.py 14.mp4
```

此步骤读取已计算的 `transforms_homography.npz`，逐帧排除与画面外部连通的原有黑边，
为 Lanczos4 插值保留 4 像素源图边界余量，将有效区域映射到固定参考坐标后求全部帧交集。
在交集内精确求解面积最大的轴对齐矩形，宽、高均取偶数以适配 H.264/yuv420p；
不限制原宽高比，不做缩放。这里的“最大”指上述有效区域定义和偶数尺寸约束下的最大值。

随后直接从原片使用已有变换一次重采样生成 `14_stabilized_cropped.mp4`，
避免对先前稳像成品再次转码。整段采用同一裁切框，因此静止物体不会因裁切而逐帧移动。
坐标仅发生固定偏移：`x_new = x_stabilized - crop_x`，`y_new = y_stabilized - crop_y`。

- `crop_geometry.json`：固定裁切坐标、尺寸、有效区域定义和保留面积。
- `crop_common_valid_mask.png`：所有帧有效区域交集。
- `crop_preview.jpg`：原稳定坐标画面上的裁切框预览。
- `transforms_homography_cropped.npz`：裁切坐标到原片坐标的每帧变换。
- `crop_verification.json`：全片解码、输出参数和音轨哈希验证结果。

可用 `--stage geometry`、`--stage render`、`--stage verify` 分步执行；已存在的最终视频不会被覆盖。
