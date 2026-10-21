"""Find the maximum common fixed crop, then render it directly from original frames."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from stabilize_static import ffmpeg_path, log, video_info


def largest_rectangle(mask, multiple=2):
    """Exact largest axis-aligned rectangle with width/height divisible by multiple."""
    heights = np.zeros(mask.shape[1], dtype=int)
    best = (0, 0, 0, 0)
    best_area = 0
    for y, row in enumerate(mask):
        heights = np.where(row, heights+1, 0)
        stack = []
        for x in range(len(heights)+1):
            height = int(heights[x]) if x < len(heights) else 0
            start = x
            while stack and stack[-1][1] > height:
                left, old_height = stack.pop()
                width = (x-left)//multiple*multiple
                hh = old_height//multiple*multiple
                area = width*hh
                if area > best_area:
                    best_area = area
                    best = (left, y-hh+1, width, hh)
                start = left
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    if best_area == 0:
        raise RuntimeError('No common valid rectangle')
    return best


def valid_source(frame, threshold=12, margin=4):
    # Mark only black regions connected to the image exterior, preserving interior dark objects.
    black = (np.max(frame, axis=2) <= threshold).astype(np.uint8)
    padded = np.pad(black, 1, constant_values=1)
    cv2.floodFill(padded, None, (0, 0), 2)
    valid = (padded[1:-1, 1:-1] != 2).astype(np.uint8)*255
    # Lanczos4 uses an 8-sample footprint. The margin avoids mixing black-border pixels.
    return cv2.erode(valid, np.ones((margin*2+1, margin*2+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)


def geometry(source, out, matrices):
    info = video_info(source)
    w, h = info['width'], info['height']
    cap = cv2.VideoCapture(str(source))
    common = np.full((h, w), 255, np.uint8)
    for i, matrix in enumerate(matrices):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Decode failed at {i}')
        valid = valid_source(frame)
        # Linear interpolation + threshold keeps only fully supported mask samples.
        warped = cv2.warpPerspective(valid, matrix, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        common[warped < 255] = 0
        if i % 400 == 0:
            log(f'Common valid area: {i}/{len(matrices)}')
    cap.release()
    x, y, cw, ch = largest_rectangle(common > 0)
    assert np.all(common[y:y+ch, x:x+cw] == 255)
    unrestricted = largest_rectangle(common > 0, multiple=1)
    crop = dict(x=x, y=y, width=cw, height=ch, right_exclusive=x+cw, bottom_exclusive=y+ch,
                source_width=w, source_height=h, frames_checked=len(matrices),
                retained_canvas_fraction=cw*ch/(w*h),
                maximum_area_even_dimensions=cw*ch,
                unrestricted_rectangle=list(unrestricted),
                black_threshold=12, source_interpolation_margin_px=4,
                method='Intersection of all frame support masks, excluding exterior-connected '
                       'source black bars and Lanczos4 support margin. Exact maximal axis-aligned '
                       'rectangle with even width and height. No aspect ratio restriction.',
                coordinate_mapping='cropped_x = stabilized_x - x; cropped_y = stabilized_y - y')
    cv2.imwrite(str(out/'crop_common_valid_mask.png'), common)
    (out/'crop_geometry.json').write_text(json.dumps(crop, indent=2), encoding='utf-8')
    preview = cv2.imread(str(out/'background.png'), cv2.IMREAD_COLOR)
    preview[common == 0] = (preview[common == 0]*.35).astype(np.uint8)
    cv2.rectangle(preview, (x, y), (x+cw-1, y+ch-1), (20, 210, 20), 2)
    cv2.imwrite(str(out/'crop_preview.jpg'), preview)
    # Reference -> source maps change only by one constant coordinate translation.
    translation = np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])
    cropped_matrices = matrices @ translation
    np.savez_compressed(out/'transforms_homography_cropped.npz', matrices=cropped_matrices,
                        indices=np.arange(len(matrices)), crop_xy=np.array([x, y]))
    log(json.dumps(crop))
    return crop


def render(source, out, matrices, crop):
    info = video_info(source)
    x, y, w, h = [crop[k] for k in ('x', 'y', 'width', 'height')]
    target = out/f'{source.stem}_stabilized_cropped.mp4'
    partial = out/f'{source.stem}_stabilized_cropped.partial.mp4'
    if target.exists():
        raise FileExistsError(target)
    command = [ffmpeg_path(), '-hide_banner', '-loglevel', 'warning', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(info['fps']),
        '-i', '-', '-i', str(source), '-map', '0:v:0', '-map', '1:a?',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '17',
        '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-pix_fmt', 'yuv420p',
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-color_range', 'tv', '-c:a', 'copy', '-movflags', '+faststart', str(partial)]
    translation = np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])
    cap = cv2.VideoCapture(str(source))
    with (out/'crop_encoding.log').open('w', encoding='utf-8') as err:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=err)
        try:
            for i, matrix in enumerate(matrices):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Decode failed at {i}')
                result = cv2.warpPerspective(frame, matrix @ translation, (w, h),
                    flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT)
                process.stdin.write(result.tobytes())
                if i % 400 == 0:
                    log(f'Crop render: {i}/{len(matrices)}')
            process.stdin.close()
            if process.wait():
                raise RuntimeError('FFmpeg failed; see crop_encoding.log')
        except BaseException:
            process.kill(); process.wait(); raise
        finally:
            cap.release()
    expected = dict(info, width=w, height=h)
    if video_info(partial) != expected:
        raise RuntimeError('Output metadata mismatch')
    partial.replace(target)
    return target


def verify(source, target, out, crop):
    info = video_info(target)
    cap = cv2.VideoCapture(str(target))
    count = 0
    samples = {0, info['frames']//2, info['frames']-1}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count in samples:
            cv2.imwrite(str(out/f'cropped_frame_{count:04}.jpg'), frame)
        count += 1
    cap.release()
    if count != info['frames'] or count != crop['frames_checked']:
        raise RuntimeError(f'Unexpected decoded frame count {count}')
    hashes = {}
    for path in (source, target):
        run = subprocess.run([ffmpeg_path(), '-hide_banner', '-loglevel', 'error', '-i', str(path),
            '-map', '0:a:0', '-c:a', 'copy', '-f', 'hash', '-'], capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f'Audio verification failed: {run.stderr}')
        hashes[path.name] = run.stdout.strip()
    if len(set(hashes.values())) != 1:
        raise RuntimeError('Audio hash mismatch')
    result = dict(decoded_frames=count, metadata=info, audio_hashes=hashes,
                  audio_identical=True, crop=crop)
    (out/'crop_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    log(f'Verified {count} frames; audio unchanged: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', choices=['geometry', 'render', 'verify', 'all'], default='all')
    args = parser.parse_args()
    source = args.input.resolve()
    out = args.output or source.parent/f'{source.stem}_stabilized'
    data = np.load(out/'transforms_homography.npz')
    matrices = data['matrices']
    if not np.array_equal(data['indices'], np.arange(video_info(source)['frames'])):
        raise RuntimeError('Expected one transform for every original frame')
    if args.stage in ('geometry', 'all'):
        crop = geometry(source, out, matrices)
    else:
        crop = json.loads((out/'crop_geometry.json').read_text(encoding='utf-8'))
    if args.stage in ('render', 'all'):
        render(source, out, matrices, crop)
    if args.stage in ('verify', 'all'):
        verify(source, out/f'{source.stem}_stabilized_cropped.mp4', out, crop)


if __name__ == '__main__':
    main()
