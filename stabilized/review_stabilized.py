"""Verify the encoded video, draw residual plots, and export a native-scale comparison."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from stabilize_static import (checkpoint_stats, ffmpeg_path, gray, log, project,
                              read_frame, summarize, track_checkpoints, video_info)


def verify(source, out, model, source_mode=False):
    path = source if source_mode else out/f'{source.stem}_stabilized.mp4'
    prefix = 'source' if source_mode else 'encoded'
    reference = np.load(out/'reference.npz')
    template, points = reference['background'], reference['validation']
    transforms = dict(np.load(out/f'transforms_{model}.npz'))
    info = video_info(path)
    cap = cv2.VideoCapture(str(path))
    rows, deltas, availability, before = [], [], [], []
    for i in range(info['frames']):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Encoded frame {i} failed to decode')
        matrix = transforms['matrices'][i] if source_mode else np.eye(3)
        q, good = track_checkpoints(template, gray(frame), points, matrix, reference['stable'])
        stat, delta, valid, raw_delta = checkpoint_stats(points, q, good, matrix, template.shape)
        rows.append(dict(frame=i, time_s=i/info['fps'], **stat))
        deltas.append(delta); availability.append(valid); before.append(raw_delta)
        if i % 300 == 0:
            log(f'{prefix} verification {i}/{info["frames"]}')
    if cap.read()[0]:
        raise RuntimeError('Unexpected extra decoded frame')
    cap.release()
    deltas, availability, before = np.asarray(deltas), np.asarray(availability), np.asarray(before)
    norms = np.linalg.norm(deltas, axis=2)
    measured = norms[availability]
    counts = availability.sum(axis=1)
    core = ((points[:, 0] > .35*info['width']) & (points[:, 0] < .65*info['width']) &
            (points[:, 1] > .40*info['height']) & (points[:, 1] < .70*info['height']))
    core_measurements = norms[availability & core[None]]
    summary = dict(decoded_frames=len(rows), resolution=[info['width'], info['height']],
        fps=info['fps'], checkpoints=len(points), available_observations=int(availability.sum()),
        availability_fraction=float(availability.mean()), min_checkpoints_per_frame=int(counts.min()),
        after_median_px=float(np.median(measured)), after_p95_px=float(np.percentile(measured, 95)),
        after_max_px=float(measured.max()), fraction_below_half_px=float((measured < .5).mean()),
        fraction_below_one_px=float((measured < 1).mean()),
        frames_with_at_least_6_checks=int((counts >= 6).sum()),
        frames_with_p95_over_1px=sum(r['after_p95'] > 1 for r in rows),
        frames_with_any_error_over_1px=int(((norms > 1) & availability).any(axis=1).sum()),
        core_region=[.35*info['width'], .40*info['height'], .65*info['width'], .70*info['height']],
        core_checkpoint_count=int(core.sum()), core_available_observations=len(core_measurements),
        core_median_px=float(np.median(core_measurements)) if len(core_measurements) else None,
        core_p95_px=float(np.percentile(core_measurements, 95)) if len(core_measurements) else None,
        caveat='Independent template-matching estimates on visible, sufficiently textured patches. '
               'Not external ground truth; unavailable/occluded patches do not count as zero. '
               'Checks search +/-8 pixels and reject ambiguous or boundary peaks. '
               'No final-residual threshold removes observations.')
    (out/f'{prefix}_verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(out/f'{prefix}_verification.npz', deltas=deltas, valid=availability, before=before)
    with (out/f'{prefix}_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    log(json.dumps(summary))
    if source_mode:
        # Refresh audit columns using the exact matrices that were rendered; never refit here.
        with (out/f'metrics_{model}.csv').open(encoding='utf-8-sig', newline='') as f:
            fit_rows = list(csv.DictReader(f))
        for row, measured_row in zip(fit_rows, rows):
            row.update(measured_row)
            row['status'] = ('review' if row['checkpoints'] < 6 or row['after_p95'] > 1 else 'ok')
        with (out/f'metrics_{model}.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(fit_rows[0])); writer.writeheader(); writer.writerows(fit_rows)
        transforms.update(check_deltas=deltas, check_valid=availability, check_before=before)
        np.savez_compressed(out/f'transforms_{model}.npz', **transforms)
        summarize(out, model, fit_rows, deltas, availability, before)
    else:
        plot_results(out, model, info, rows, norms, availability)


def plot_results(out, model, info, rows, norms, available):
    transform = np.load(out/f'transforms_{model}.npz')
    ref = np.load(out/'reference.npz')
    points = ref['validation']
    times = np.arange(info['frames']) / info['fps']
    center = np.array([[info['width']/2, info['height']/2]], np.float32)
    movement = np.stack([project(center, m)[0]-center[0] for m in transform['matrices']])
    original_errors = np.linalg.norm(transform['check_before'], axis=2)
    raw = original_errors[transform['check_valid']]
    fixed = norms[available]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), layout='constrained')
    ax[0].plot(times, movement[:, 0], label='Original camera displacement: X', lw=1.4)
    ax[0].plot(times, movement[:, 1], label='Original camera displacement: Y', lw=1.4)
    ax[0].set(ylabel='Pixels at image center', title='Fixed-reference video stabilization')
    ax[0].legend(loc='upper right'); ax[0].grid(alpha=.2)
    ax[1].plot(times, [r['after_median'] for r in rows], label='Encoded output: median', lw=.9)
    ax[1].plot(times, [r['after_p95'] for r in rows], label='Encoded output: 95th percentile', lw=.6, alpha=.7)
    ax[1].axhline(.5, color='green', ls='--', lw=.8, label='0.5 pixel')
    ax[1].axhline(1., color='orange', ls='--', lw=.8, label='1 pixel')
    ax[1].set(ylabel='Checkpoint residual (pixels)', xlabel='Time (seconds)',
              ylim=(0, max(2., np.nanmax([r['after_p95'] for r in rows])*1.1)))
    ax[1].legend(loc='upper right', ncols=2); ax[1].grid(alpha=.2)
    ax[2].plot(times, available.sum(axis=1), color='#357a76', lw=.6)
    ax[2].set(ylabel='Available checkpoints', xlabel='Time (seconds)', ylim=(0, len(points)))
    ax[2].grid(alpha=.2)
    fig.savefig(out/'residuals.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for values, name in [(raw, 'Original'), (fixed, 'Stabilized (encoded)')]:
        values = np.sort(values)
        ax.plot(values, np.arange(1, len(values)+1)/len(values)*100, label=name)
    ax.set(xlabel='Checkpoint displacement (pixels)', ylabel='Cumulative observations (%)',
           xlim=(0, max(2., np.percentile(raw, 99))), ylim=(0, 100),
           title='Visible independent checkpoints; occlusions excluded by appearance checks')
    ax.axvline(.5, color='gray', lw=.8, ls='--'); ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'error_distribution.png', dpi=150); plt.close(fig)


def comparison(source, out, model):
    info = video_info(source)
    original = cv2.VideoCapture(str(source))
    corrected = cv2.VideoCapture(str(out/f'{source.stem}_stabilized.mp4'))
    # Native-resolution central crop; the comparison contains only labels and fixed crosshairs.
    cw = min(1280, info['width']); ch = min(480, info['height'])
    x0, y0 = (info['width']-cw)//2, (info['height']-ch)//2
    ref = np.load(out/'reference.npz')
    checkpoints = ref['validation']
    starts = [0, max(0, info['frames']//2-75), max(0, info['frames']-150)]
    target = out/f'{source.stem}_center_comparison.mp4'
    cmd = [ffmpeg_path(), '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'rawvideo',
           '-pix_fmt', 'bgr24', '-s', f'{cw*2}x{ch+50}', '-r', str(info['fps']), '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '17', '-pix_fmt', 'yuv420p',
           '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-colorspace', 'bt709',
           '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
           '-movflags', '+faststart', str(target)]
    with (out/'comparison_encoding.log').open('w') as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        try:
            for start in starts:
                original.set(cv2.CAP_PROP_POS_FRAMES, start)
                corrected.set(cv2.CAP_PROP_POS_FRAMES, start)
                for i in range(start, min(start+150, info['frames'])):
                    ok1, a = original.read(); ok2, b = corrected.read()
                    if not ok1 or not ok2:
                        raise RuntimeError('Comparison frame decode failed')
                    canvas = np.zeros((ch+50, cw*2, 3), np.uint8)
                    for k, (frame, label) in enumerate([(a, 'ORIGINAL'), (b, 'STABILIZED')]):
                        tile = frame[y0:y0+ch, x0:x0+cw].copy()
                        # Fixed locations, not tracked overlays: movement remains visible.
                        for px, py in checkpoints:
                            x, y = int(px-x0), int(py-y0)
                            if 10 < x < cw-10 and 10 < y < ch-10:
                                for dx, dy in [(-9, 0), (9, 0), (0, -9), (0, 9)]:
                                    cv2.line(tile, (x+dx, y+dy), (x+int(dx*1.7), y+int(dy*1.7)), (255, 160, 0), 1)
                        canvas[50:, k*cw:(k+1)*cw] = tile
                        cv2.putText(canvas, f'{label}   t={i/info["fps"]:.2f}s   1:1 center crop',
                                    (k*cw+15, 32), 0, .8, (240, 240, 240), 1, cv2.LINE_AA)
                    proc.stdin.write(canvas.tobytes())
                    if i == start:
                        cv2.imwrite(str(out/f'comparison_frame_{i:04}.jpg'), canvas)
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError('Comparison encoding failed')
        except BaseException:
            proc.kill(); proc.wait(); raise
        finally:
            original.release(); corrected.release()
    log(f'Comparison written: {target}')


def report(source, out, model):
    info = video_info(source)
    estimated = json.loads((out/f'summary_{model}.json').read_text(encoding='utf-8'))
    encoded = json.loads((out/'encoded_verification.json').read_text(encoding='utf-8'))
    ref = np.load(out/'reference.npz')
    audio_path = out/'audio_verification.json'
    audio_data = json.loads(audio_path.read_text(encoding='utf-8')) if audio_path.exists() else {}
    audio_note = ('原音轨直接复制，音轨 SHA-256 校验一致。' if len(audio_data) == 2 and
                  len(set(audio_data.values())) == 1 else '原音轨按输入直接复制；未单独完成音轨哈希校验。')
    models = []
    for name in ('similarity', 'affine', 'homography'):
        path = out/f'summary_{name}_sample.json'
        if path.exists():
            s = json.loads(path.read_text(encoding='utf-8'))
            models.append(f"| {name} | {s['after_median_px']:.3f} | {s['after_p95_px']:.3f} | {s['checkpoint_observations']} |")
    fewer = info['frames']-encoded['frames_with_at_least_6_checks']
    core_text = (f"中心更小区域（x={encoded['core_region'][0]:.0f}～{encoded['core_region'][2]:.0f}，"
                 f"y={encoded['core_region'][1]:.0f}～{encoded['core_region'][3]:.0f}）共有 "
                 f"{encoded['core_checkpoint_count']} 个候选检查点，累计 "
                 f"{encoded['core_available_observations']} 次有效观测，"
                 f"误差中位数 {encoded['core_median_px']:.3f} 像素，"
                 f"95% 分位数 {encoded['core_p95_px']:.3f} 像素。"
                 if encoded['core_median_px'] is not None else '中心更小区域缺少可靠观测，未作精度承诺。')
    text = f'''# {source.name} 中心静态背景稳像结果

已输出完整视频：[{source.stem}_stabilized.mp4]({source.stem}_stabilized.mp4)。
先看 [15 秒中心前后对比]({source.stem}_center_comparison.mp4)：左侧原片、右侧稳像，
依次展示开头、中间和结尾，每段 5 秒；中心裁切保持 1:1 像素，蓝色标记固定在参考坐标。

## 实际结果

| 项目 | 结果 |
|---|---:|
| 分辨率 | {info['width']} × {info['height']} |
| 帧率 / 帧数 / 时长 | {info['fps']:g} fps / {info['frames']} 帧 / {info['frames']/info['fps']:.1f} 秒 |
| 原片检查点位移中位数 | {estimated['before_median_px']:.3f} 像素 |
| 原片检查点位移 95% 分位数 | {estimated['before_p95_px']:.3f} 像素 |
| 成品残余误差中位数 | {encoded['after_median_px']:.3f} 像素 |
| 成品残余误差 95% 分位数 | {encoded['after_p95_px']:.3f} 像素 |
| 成品有效观测误差 < 0.5 像素 | {encoded['fraction_below_half_px']:.2%} |
| 成品有效观测误差 < 1 像素 | {encoded['fraction_below_one_px']:.2%} |
| 成品最大单次测量误差（保留异常） | {encoded['after_max_px']:.3f} 像素 |
| 成品全部帧解码检查 | {encoded['decoded_frames']} 帧通过 |
| 每帧至少 6 个有效检查点 | {encoded['frames_with_at_least_6_checks']} / {info['frames']} 帧 |
| 每帧可测点数最少值 | {encoded['min_checkpoints_per_frame']} |

误差单位对应原始分辨率。上表的分位数合并了多个检查点、多个时刻的**有效观测**，
不是保证每一帧、每一个像素都小于这个数。原片与成品的可测观测集合可能略有不同。
测量方法是图像匹配，没有外部坐标真值，不能等同于测量仪器认证精度。

{core_text}

## 独立检查与异常

用于估计相机变换的参考点有 {len(ref['training'])} 个；另外保留 {len(ref['validation'])} 个检查点，
不参与变换拟合，并排除了其附近的拟合特征。检查点覆盖中心 50% 宽度、纵向 15%～90% 区域。
成品共得到 {encoded['available_observations']} 次有效观测，占全部候选点帧组合的
{encoded['availability_fraction']:.2%}。这反映了人车遮挡、局部纹理不足及相似性检查的影响。

验收使用带时间稳定性遮罩的局部归一化相关匹配和亚像素 ECC。
遮罩来自处理前抽样构建的背景变化统计；它排除会摆动的隔离带等像素，
不依据最终误差大小筛选。搜索范围为预测位置周围 ±8 像素；
低相关、匹配不唯一、峰值落在搜索边界或纹理不足时标记不可测，不记为零误差。

成品有 {fewer} 帧的有效检查点不足 6 个；
有 {encoded['frames_with_p95_over_1px']} 帧的有效点误差 95% 分位数超过 1 像素；
有 {encoded['frames_with_any_error_over_1px']} 帧至少一个有效点误差超过 1 像素。
这些记录均保留在 [成品逐帧检查](encoded_metrics.csv) 中，未删帧或强行替换背景。
个别大误差仍可能包含部分遮挡、局部外观改变或误匹配，不能仅凭单个检查点判定整帧相机偏移。

![逐帧误差与有效检查点数量](residuals.png)

![原片与成品误差分布](error_distribution.png)

## 处理方式

从整段视频抽取 {len(ref['sample_indices'])} 帧，进行静态背景对齐及时间中位数建模；
逐帧用分布在路面、标线和周边铺装上的静态特征向固定背景配准，
以中心加权的稳健单应性变换精修。前一帧结果只用于初始化，不逐帧累乘相机位移。
参考背景只用于运动估计，成品的每一帧都来自对应原始帧的一次重采样。

原始帧数、帧率和分辨率保持一致；不做动态裁剪或动态缩放，边缘空缺填黑。
编码为 H.264、CRF 17，BT.709。{audio_note}
原始文件未被修改。

悬空隔离带、行人和车辆的自身运动不会被消除。单个平面变换主要稳定地面，
不同高度物体的视差、阴影变化、遮挡不能保证同时固定。

## 抽样模型比较

每 60 帧取一帧，共 49 帧；使用相同的独立检查方法。模型间可测点数有差异，
所以这些数字用于辅助模型选择，不是严格相同观测集合上的算法基准。

| 模型 | 残差中位数 / 像素 | 95% 分位数 / 像素 | 有效观测数 |
|---|---:|---:|---:|
{chr(10).join(models)}

合成验证使用已知亚像素位移、旋转、透视变化及人工遮挡，
三种测试中的最大相机变换恢复误差均小于 0.05 像素。
这是算法实现方向与数值精度检查，不代表真实视频所有帧也能达到该值。

可复用脚本和运行命令见上一级目录的 `STABILIZATION_README.md`。
每段视频需分别构建参考背景，不能共用 14.mp4 的变换矩阵。
'''
    (out/'处理报告.md').write_text(text, encoding='utf-8')
    log(f'Report written: {out / "处理报告.md"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--model', default='homography')
    p.add_argument('--stage', choices=['source', 'verify', 'comparison', 'report', 'all'], default='all')
    args = p.parse_args(); source = args.input.resolve()
    out = args.output or source.parent/f'{source.stem}_stabilized'
    if args.stage in ('source', 'all'):
        verify(source, out, args.model, source_mode=True)
    if args.stage in ('verify', 'all'):
        verify(source, out, args.model)
    if args.stage in ('comparison', 'all'):
        comparison(source, out, args.model)
    if args.stage in ('report', 'all'):
        report(source, out, args.model)


if __name__ == '__main__':
    main()
