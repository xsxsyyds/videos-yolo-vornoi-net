"""Register hovering-camera videos to a fixed static-background coordinate system.

Original frames are never replaced by the background: it is an alignment template only.
Run --help for the three stages (prepare, estimate, render) and resumable outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

cv2.setNumThreads(6)


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def project(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2),
                                    np.asarray(matrix, np.float64)).reshape(-1, 2)


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    info = dict(width=int(cap.get(3)), height=int(cap.get(4)),
                fps=float(cap.get(5)), frames=int(cap.get(7)))
    cap.release()
    return info


def read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot decode frame {index}")
    return frame


def sift_homography(ref, cur, max_width=1600):
    scale = min(1., max_width / ref.shape[1])
    a = cv2.resize(gray(ref), None, fx=scale, fy=scale)
    b = cv2.resize(gray(cur), (a.shape[1], a.shape[0]))
    sift = cv2.SIFT_create(nfeatures=6500, contrastThreshold=.012, edgeThreshold=12)
    ka, da = sift.detectAndCompute(a, (a > 12).astype(np.uint8) * 255)
    kb, db = sift.detectAndCompute(b, (b > 12).astype(np.uint8) * 255)
    if da is None or db is None:
        raise RuntimeError("Insufficient SIFT descriptors")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    matches = [m for pair in matcher.knnMatch(da, db, k=2) if len(pair) == 2
               for m, n in [pair] if m.distance < .72 * n.distance]
    p = np.float32([ka[m.queryIdx].pt for m in matches]) / scale
    q = np.float32([kb[m.trainIdx].pt for m in matches]) / scale
    if len(p) < 15:
        raise RuntimeError("Insufficient SIFT matches")
    matrix, inliers = cv2.findHomography(p, q, cv2.USAC_MAGSAC, 2.0,
                                       maxIters=5000, confidence=.999)
    if matrix is None or int(inliers.sum()) < 12:
        raise RuntimeError("SIFT registration failed")
    return matrix, int(inliers.sum())


def make_background(frames, matrices, shape):
    h, w = shape
    aligned = np.stack([cv2.warpPerspective(gray(f), m, (w, h),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                        for f, m in zip(frames, matrices)])
    # Tile statistics to avoid several full-size floating-point sample stacks.
    background = np.zeros((h, w), np.uint8)
    variability = np.zeros((h, w), np.float32)
    for y in range(0, h, 80):
        block = aligned[:, y:y+80].astype(np.float32)
        med = np.median(block, axis=0)
        background[y:y+80] = np.rint(med).astype(np.uint8)
        variability[y:y+80] = np.percentile(np.abs(block-med), 65, axis=0)
    stable = ((variability < 10) & (background > 18)).astype(np.uint8) * 255
    stable = cv2.erode(stable, np.ones((7, 7), np.uint8))
    stable[:20] = 0; stable[-20:] = 0; stable[:, :20] = 0; stable[:, -20:] = 0
    return background, variability, stable


def select_points(background, stable):
    h, w = background.shape
    # Spatial distribution avoids allowing a crowded or richly textured corner to dominate.
    points = []
    for yi in range(5):
        for xi in range(16):
            x0, x1 = xi*w//16, (xi+1)*w//16
            y0, y1 = yi*h//5, (yi+1)*h//5
            p = cv2.goodFeaturesToTrack(background[y0:y1, x0:x1], 24, .008, 11,
                                       mask=stable[y0:y1, x0:x1], blockSize=5)
            if p is not None:
                points.extend(p.reshape(-1, 2) + [x0, y0])
    points = np.asarray(points, np.float32)
    # Held-out checkpoints never participate in estimating the transformation.
    quality = cv2.cornerMinEigenVal(background, 5)
    candidates = []
    for p in points:
        x, y = map(int, p)
        if .25*w < x < .75*w and .15*h < y < .90*h:
            patch = background[y-20:y+21, x-20:x+21]
            mask = stable[y-20:y+21, x-20:x+21]
            if patch.shape == (41, 41) and patch.std() > 6 and (mask > 0).mean() > .65:
                candidates.append((float(quality[y, x]), p))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    validation = []
    for low, high, quota in [(.30, .70, 16), (.15, .30, 20), (.70, .90, 20)]:
        selected = 0
        for _, p in candidates:
            if low*h <= p[1] < high*h and all(np.linalg.norm(p-v) > 38 for v in validation):
                validation.append(p)
                selected += 1
                if selected >= quota:
                    break
    validation = np.asarray(validation, np.float32).reshape(-1, 2)
    if len(validation) < 6:
        raise RuntimeError("Too few independent central validation points")
    distances = np.linalg.norm(points[:, None] - validation[None], axis=2)
    training = points[np.min(distances, axis=1) > 34]
    if len(training) < 40:
        raise RuntimeError("Too few stable reference features")
    return training, validation


def prepare(args, out):
    info = video_info(args.input)
    cap = cv2.VideoCapture(str(args.input))
    indices = np.unique(np.linspace(0, info['frames']-1, args.samples, dtype=int))
    frames = [read_frame(cap, i) for i in indices]
    cap.release()
    reference = frames[0]
    matrices = []
    for i, frame in zip(indices, frames):
        matrix, n = (np.eye(3), 0) if i == 0 else sift_homography(reference, frame)
        matrices.append(matrix)
        log(f"Background sample {i}: {n} static-consensus matches")
    background, variability, stable = make_background(frames, matrices, reference.shape[:2])
    # One fixed-reference refinement makes the temporal template less dependent on frame 0's people.
    for j, frame in enumerate(frames):
        matrices[j], _ = sift_homography(background, frame)
    background, variability, stable = make_background(frames, matrices, reference.shape[:2])
    training, validation = select_points(background, stable)
    np.savez_compressed(out/'reference.npz', background=background, stable=stable,
                        variability=variability, training=training, validation=validation,
                        sample_indices=indices, sample_matrices=np.stack(matrices))
    cv2.imwrite(str(out/'background.png'), background)
    preview = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    preview[stable == 0] = (preview[stable == 0] * .45).astype(np.uint8)
    for x, y in training:
        cv2.circle(preview, (round(float(x)), round(float(y))), 2, (60, 210, 60), -1)
    for k, (x, y) in enumerate(validation):
        cv2.circle(preview, (round(float(x)), round(float(y))), 6, (0, 150, 255), 2)
        cv2.putText(preview, str(k), (int(x)+5, int(y)-5), 0, .35, (0, 150, 255), 1)
    cv2.imwrite(str(out/'reference_points.jpg'), preview)
    (out/'video_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    log(f"Reference ready: {len(training)} fit points; {len(validation)} held-out checkpoints")


def track(template, current, points, guess):
    p = points.reshape(-1, 1, 2).astype(np.float32)
    predicted = project(points, guess).reshape(-1, 1, 2)
    q, status, error = cv2.calcOpticalFlowPyrLK(
        template, current, p, predicted, winSize=(25, 25), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 35, .005),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=.00008)
    back, status2, _ = cv2.calcOpticalFlowPyrLK(
        current, template, q, p.copy(), winSize=(25, 25), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .005),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=.00008)
    fb = np.linalg.norm(back-p, axis=2).ravel()
    good = (status.ravel() > 0) & (status2.ravel() > 0) & (fb < .45) & (error.ravel() < 13)
    return q.reshape(-1, 2), good, error.ravel(), fb


def track_checkpoints(template, current, points, matrix, stable=None):
    """Independent local NCC + translation ECC; no fitting-point flow reused here.

    Search +/- 8 pixels around the model prediction. Ambiguous, occluded and
    boundary-peak patches are reported unavailable, never as zero-error measurements.
    """
    predicted = project(points, matrix)
    observed = predicted.copy()
    valid = np.zeros(len(points), bool)
    for k, (point, prediction) in enumerate(zip(points, predicted)):
        patch = cv2.getRectSubPix(template, (41, 41), tuple(map(float, point)))
        search = cv2.getRectSubPix(current, (57, 57), tuple(map(float, prediction)))
        # A patch may contain a moving ribbon even though its center is on static asphalt.
        # Use the independently constructed temporal stability mask, not the measured residual,
        # to exclude those pixels. Featureless masked patches are explicitly unavailable.
        mask = np.ones((41, 41), np.float32) if stable is None else (
            cv2.getRectSubPix(stable, (41, 41), tuple(map(float, point))) > 250).astype(np.float32)
        count = float(mask.sum())
        if count < 41*41*.45:
            continue
        mean = float((patch*mask).sum()/count)
        zero_mean = (patch.astype(np.float32)-mean)*mask
        energy = float((zero_mean**2).sum())
        if energy/count < 4.**2:
            continue
        values = search.astype(np.float32)
        numerator = cv2.matchTemplate(values, zero_mean, cv2.TM_CCORR)
        total = cv2.matchTemplate(values, mask, cv2.TM_CCORR)
        squared = cv2.matchTemplate(values**2, mask, cv2.TM_CCORR)
        denominator = np.sqrt(np.maximum(energy*(squared-total**2/count), 1.e-6))
        response = numerator/denominator
        _, peak, _, (x, y) = cv2.minMaxLoc(response)
        if peak < .90 or x in (0, 16) or y in (0, 16):
            continue
        other = response.copy()
        other[max(0, y-3):y+4, max(0, x-3):x+4] = -1
        if peak-float(other.max()) < .015:
            continue
        center = prediction + [x-8, y-8]
        candidate = cv2.getRectSubPix(current, (41, 41), tuple(map(float, center)))
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            score, warp = cv2.findTransformECC(patch, candidate, warp, cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .00001),
                (mask*255).astype(np.uint8), 3)
        except cv2.error:
            continue
        if score > .93 and np.linalg.norm(warp[:, 2]) < 2.:
            observed[k] = center + warp[:, 2]
            valid[k] = True
    return observed, valid


def fit_transform(p, q, shape, model='homography'):
    if len(p) < 20:
        raise RuntimeError(f"Only {len(p)} usable tracks")
    h, w = shape
    if model == 'similarity':
        m, inside = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
                                               ransacReprojThreshold=1., maxIters=3000,
                                               confidence=.999, refineIters=20)
        matrix = None if m is None else np.vstack([m, [0, 0, 1]])
    elif model == 'affine':
        m, inside = cv2.estimateAffine2D(p, q, method=cv2.RANSAC,
                                        ransacReprojThreshold=1., maxIters=3000,
                                        confidence=.999, refineIters=20)
        matrix = None if m is None else np.vstack([m, [0, 0, 1]])
    else:
        matrix, inside = cv2.findHomography(p, q, cv2.USAC_MAGSAC, 1.,
                                           maxIters=3000, confidence=.999)
    if matrix is None or int(inside.sum()) < 20:
        raise RuntimeError("No robust static-point consensus")
    inside = inside.ravel().astype(bool)
    pp, qq = p[inside], q[inside]
    # Central observations carry more weight, but distant references still constrain rotation.
    weights = .25 + .75*np.exp(-.5*(((pp[:, 0]-w/2)/(.27*w))**2 +
                                   ((pp[:, 1]-h/2)/(.40*h))**2))
    normalizer = np.array([[1/w, 0, -.5], [0, 1/w, -h/(2*w)], [0, 0, 1.]])
    inv_normalizer = np.linalg.inv(normalizer)
    pn = project(pp, normalizer).astype(np.float64)
    qn = project(qq, normalizer).astype(np.float64)
    mn = normalizer @ matrix @ inv_normalizer
    mn /= mn[2, 2]

    def unpack(v):
        if model == 'similarity':
            return np.array([[v[0], -v[1], v[2]], [v[1], v[0], v[3]], [0, 0, 1.]])
        if model == 'affine':
            return np.vstack([v.reshape(2, 3), [0, 0, 1.]])
        return np.r_[v, 1.].reshape(3, 3)

    if model == 'similarity':
        v0 = [mn[0, 0], mn[1, 0], mn[0, 2], mn[1, 2]]
    elif model == 'affine':
        v0 = mn[:2].ravel()
    else:
        v0 = mn.ravel()[:8]

    def residual(v):
        mm = unpack(v)
        xy = pn @ mm[:2, :2].T + mm[:2, 2]
        xy /= (pn @ mm[2, :2] + mm[2, 2])[:, None]
        return ((xy-qn)*w*np.sqrt(weights[:, None])).ravel()

    fit = least_squares(residual, v0, loss='soft_l1', f_scale=.25, max_nfev=35,
                        ftol=1e-7, xtol=1e-8, gtol=1e-7)
    matrix = inv_normalizer @ unpack(fit.x) @ normalizer
    matrix /= matrix[2, 2]
    errors = np.linalg.norm(project(pp, matrix)-qq, axis=1)
    return matrix, int(inside.sum()), float(np.median(errors))


def checkpoint_stats(points, observed, good, matrix, shape):
    h, w = shape
    center = (np.abs(points[:, 0]-w/2) < .25*w) & (points[:, 1] > .15*h) & (points[:, 1] < .9*h)
    valid = good & center
    aligned = project(observed, np.linalg.inv(matrix))
    delta = aligned-points
    before = observed-points
    # Do not reject by residual magnitude: that would hide real registration failures.
    accepted = valid
    norms = np.linalg.norm(delta[accepted], axis=1)
    raw = np.linalg.norm(before[accepted], axis=1)
    result = dict(checkpoints=int(accepted.sum()), checkpoint_candidates=int(valid.sum()),
                  before_median=float(np.median(raw)) if len(raw) else math.nan,
                  after_median=float(np.median(norms)) if len(norms) else math.nan,
                  after_p95=float(np.percentile(norms, 95)) if len(norms) else math.nan,
                  after_max=float(np.max(norms)) if len(norms) else math.nan)
    return result, delta, accepted, before


def estimate(args, out):
    ref = np.load(out/'reference.npz')
    template, points, check = ref['background'], ref['training'], ref['validation']
    info = video_info(args.input)
    cap = cv2.VideoCapture(str(args.input))
    indices = list(range(0, info['frames'], args.stride))
    if args.limit:
        indices = indices[:args.limit]
    matrices, rows, check_deltas, check_valid, check_before = [], [], [], [], []
    previous = np.eye(3)
    start = time.monotonic()
    for j, i in enumerate(indices):
        if args.stride == 1:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {i}")
        else:
            frame = read_frame(cap, i)
        current = gray(frame)
        status = 'ok'
        try:
            q, good, _, _ = track(template, current, points, previous)
            matrix, inliers, fit_error = fit_transform(points[good], q[good], template.shape, args.model)
            # Reacquire from SIFT if large motion or poor coverage defeats initialized LK.
            if inliers < 45:
                raise RuntimeError("Insufficient reliable static support")
        except (cv2.error, RuntimeError):
            initial, _ = sift_homography(template, current)
            q, good, _, _ = track(template, current, points, initial)
            matrix, inliers, fit_error = fit_transform(points[good], q[good], template.shape, args.model)
            status = 'reacquired'
        cq, cg = track_checkpoints(template, current, check, matrix, ref['stable'])
        stats, delta, valid, before = checkpoint_stats(check, cq, cg, matrix, template.shape)
        # Quality gates flag uncertainty instead of silently treating a held transform as success.
        if stats['checkpoints'] < 6 or stats['after_p95'] > 1.:
            status = 'review'
        row = dict(frame=i, time_s=i/info['fps'], status=status, fit_points=inliers,
                   fit_median=fit_error, **stats)
        matrices.append(matrix); rows.append(row)
        check_deltas.append(delta); check_valid.append(valid); check_before.append(before)
        previous = matrix
        if j % 100 == 0 or j == len(indices)-1:
            log(f"{args.model}: {j+1}/{len(indices)} frame={i}, fit={inliers}, "
                f"central p95={stats['after_p95']:.3f}px, {time.monotonic()-start:.1f}s")
    cap.release()
    tag = args.model if args.stride == 1 and not args.limit else f'{args.model}_sample'
    np.savez_compressed(out/f'transforms_{tag}.npz', matrices=np.stack(matrices),
                        indices=indices, check_deltas=np.stack(check_deltas),
                        check_valid=np.stack(check_valid), check_before=np.stack(check_before))
    with (out/f'metrics_{tag}.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summarize(out, tag, rows, np.asarray(check_deltas), np.asarray(check_valid), np.asarray(check_before))


def summarize(out, tag, rows, deltas, valid, before):
    norms = np.linalg.norm(deltas, axis=2)[valid]
    raw = np.linalg.norm(before, axis=2)[valid]
    result = dict(model=tag, frames=len(rows), flagged_frames=sum(r['status']=='review' for r in rows),
                  checkpoint_observations=len(norms), before_median_px=float(np.median(raw)),
                  before_p95_px=float(np.percentile(raw, 95)), after_median_px=float(np.median(norms)),
                  after_p95_px=float(np.percentile(norms, 95)), after_max_px=float(np.max(norms)),
                  fraction_below_half_px=float(np.mean(norms < .5)),
                  fraction_below_one_px=float(np.mean(norms < 1.)),
                  min_checkpoints_per_frame=min(r['checkpoints'] for r in rows),
                  validation_note='Held-out background points in central 50% of image width, y=15..90%. '
                  'Independent local NCC and translation ECC use +/-8px search around predicted positions. '
                  'Low-correlation, ambiguous and search-boundary peaks are unavailable. '
                  'No checkpoint is excluded on the basis of its final registration residual. '
                  'This is estimated alignment error, not external ground truth.')
    (out/f'summary_{tag}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    log(json.dumps(result))


def ffmpeg_path():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def render(args, out):
    info = video_info(args.input)
    data = np.load(out/f'transforms_{args.model}.npz')
    if not np.array_equal(data['indices'], np.arange(info['frames'])):
        raise RuntimeError('Full-length transforms required for rendering')
    matrices = data['matrices']
    target = out/f'{args.input.stem}_stabilized.mp4'
    partial = out/f'{args.input.stem}_stabilized.partial.mp4'
    if target.exists():
        raise FileExistsError(f'Output already exists: {target}')
    command = [ffmpeg_path(), '-hide_banner', '-loglevel', 'warning', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f"{info['width']}x{info['height']}",
               '-r', str(info['fps']), '-i', '-', '-i', str(args.input),
               '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'fast',
               '-crf', '17', '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
               '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
               '-color_trc', 'bt709', '-color_range', 'tv',
               '-c:a', 'copy', '-movflags', '+faststart',
               str(partial)]
    cap = cv2.VideoCapture(str(args.input))
    with (out/'encoding.log').open('w', encoding='utf-8') as err:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=err)
        try:
            for i, matrix in enumerate(matrices):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Decode failed: {i}')
                corrected = cv2.warpPerspective(frame, matrix, (info['width'], info['height']),
                    flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT)
                process.stdin.write(corrected.tobytes())
                if i % 300 == 0:
                    log(f'Render {i}/{len(matrices)}')
            process.stdin.close()
            code = process.wait()
            if code:
                raise RuntimeError(f'FFmpeg failed ({code}); see encoding.log')
        except BaseException:
            process.kill(); process.wait(); raise
        finally:
            cap.release()
    actual = video_info(partial)
    if actual != info:
        raise RuntimeError(f'Output metadata mismatch: {actual} vs {info}')
    partial.replace(target)
    log(f'Written {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--stage', choices=['prepare', 'estimate', 'render', 'all'], default='all')
    parser.add_argument('--samples', type=int, default=41)
    parser.add_argument('--model', choices=['similarity', 'affine', 'homography'], default='homography')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    args.input = args.input.resolve()
    out = args.output or args.input.parent / f'{args.input.stem}_stabilized'
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ('prepare', 'all'):
        prepare(args, out)
    if args.stage in ('estimate', 'all'):
        estimate(args, out)
    if args.stage in ('render', 'all'):
        render(args, out)


if __name__ == '__main__':
    main()
