"""Render the complete annotated input with the agreed 64-vertex diagram."""
from pathlib import Path
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import pickle
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE / 'full-14-10.20'
sys.path.insert(0, str(HERE / 'deps'))
import cv2
import numpy as np
import imageio_ffmpeg
from threadpoolctl import threadpool_limits
import topology_voronoi as topology
import overlap_chords

SOURCE = HERE.parent / '14-10.20.mp4'
DEST = OUT / '14-10.20_voronoi64_polyline.mp4'
SAMPLES = (0, 500, 1400, 2400, 2850, 2900)


def compose(frame, item):
    started = time.perf_counter()
    polys = item['polys']
    assert polys.shape[1:] == (64, 2) and np.isfinite(polys).all()
    h, w = frame.shape[:2]
    if not len(polys):
        return frame, dict(agents=0, compute_compose_ms=0), {}
    F, owner, lines, pairs, stats = topology.compute(polys, (0, 0, w, h), scale=.5, repair_invalid=True)
    mask, occupied = topology.rasterize(F, polys, lines, ss=2, thick_img=2)
    assert mask.shape == (h, w)
    # Independently check that every required open-polyline segment survives
    # exterior clipping. Do this before video compression changes pixel values.
    required = np.zeros_like(mask)
    required = overlap_chords.draw_chords(F, required, F['overlap_chords'], ss=2, thick_img=2)
    missing = int(np.count_nonzero((required > 0) & (mask == 0)))
    assert missing == 0
    result = frame.copy()
    result[mask > 0] = 255
    detail = overlap_chords.serializable(F['overlap_chords'], F['overlap_issues'])
    detail['geometry_repairs'] = F['polygon_repairs']
    for group in detail.values():
        for entry in group:
            for key in ('a', 'b'):
                if key in entry:
                    i = entry[key]
                    entry[key + '_tid'] = int(item['tid'][i])
                    entry[key + '_class'] = int(item['cls'][i])
                    entry[key + '_number'] = int(item['num'][i])
    stats.update(agents=len(polys), vertices_per_mask=64,
                 segments=len(lines), required_polyline_missing_pixels=missing,
                 polyline_segments=sum(p['segment_count'] for p in F['overlap_chords']),
                 tree_ms=F['tree_ms'], query_ms=F['query_ms'],
                 mean_edges_tested=F['mean_edges_tested'],
                 compute_compose_ms=(time.perf_counter()-started)*1000)
    return result, stats, detail


def worker_init():
    cv2.setNumThreads(1)
    threadpool_limits(1)


def ordered_frames(cap, items, workers):
    # Bounded submission avoids queuing the whole decoded video in memory.
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        pending = collections.deque()
        for fi, item in enumerate(items):
            ok, frame = cap.read()
            assert ok, f'Input ended at frame {fi}'
            pending.append((fi, pool.submit(compose, frame, item)))
            if len(pending) >= workers * 2:
                index, future = pending.popleft()
                yield index, future.result()
        while pending:
            index, future = pending.popleft()
            yield index, future.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    cv2.setNumThreads(1)
    threadpool_limits(1)
    data = pickle.load((OUT / 'frames64.pkl').open('rb'))
    w, h, fps, nf, _, _ = data['meta']
    assert len(data['frames']) == nf
    cap = cv2.VideoCapture(str(SOURCE))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == nf
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (w, h)
    assert abs(cap.get(cv2.CAP_PROP_FPS) - fps) < 1e-6
    if args.preview:
        rows = []
        for fi in SAMPLES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            assert ok, fi
            result, stats, detail = compose(frame, data['frames'][fi])
            cv2.imwrite(str(OUT / f'preview_f{fi:04d}.png'), result)
            rows.append(dict(frame=fi, **stats, **detail))
            print(json.dumps(dict(frame=fi, **stats)), flush=True)
        (OUT / 'preview_stats.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        cap.release()
        return

    partial = DEST.with_suffix('.partial.mp4')
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
               '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '18', '-threads', '4',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
    totals = collections.Counter()
    issues = collections.Counter()
    timings = []
    start = time.perf_counter()
    with (OUT / 'encoder.log').open('wb') as enc_log, (OUT / 'frames.jsonl').open('w', encoding='utf-8', buffering=1) as records:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=enc_log)
        try:
            for fi, (result, stats, detail) in ordered_frames(cap, data['frames'], args.workers):
                encoder.stdin.write(result.tobytes())
                records.write(json.dumps(dict(frame=fi, seconds=fi/fps, **stats, **detail))+'\n')
                timings.append(stats['compute_compose_ms'])
                for key in ('failed_junctions', 'unbracketed_crossings', 'overlap_chord_count', 'overlap_issue_count', 'polyline_segments', 'repaired_boolean_geometries'):
                    totals[key] += stats.get(key, 0)
                for entry in detail.get('unresolved', []):
                    issues[entry['status']] += 1
                if detail.get('unresolved'):
                    totals['frames_with_overlap_issues'] += 1
                if fi in SAMPLES:
                    cv2.imwrite(str(OUT / f'result_f{fi:04d}.png'), result)
                if fi % 100 == 0 or fi == nf-1:
                    elapsed = time.perf_counter()-start
                    print(json.dumps(dict(done=fi+1, total=nf, elapsed_seconds=round(elapsed, 1),
                        eta_seconds=round(elapsed/(fi+1)*(nf-fi-1), 1),
                        mean_compute_ms=round(float(np.mean(timings)), 2), issues=dict(issues))), flush=True)
            assert not cap.read()[0], 'Unexpected extra input frames'
            encoder.stdin.close()
            assert encoder.wait() == 0, 'Encoding failed: see encoder.log'
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise
        finally:
            cap.release()
    render_seconds = time.perf_counter()-start
    # Read every output frame to catch truncation or decode errors.
    check = cv2.VideoCapture(str(partial))
    output_fps = check.get(cv2.CAP_PROP_FPS)
    decoded = 0
    while True:
        ok, frame = check.read()
        if not ok:
            break
        assert frame.shape == (h, w, 3)
        decoded += 1
    check.release()
    assert decoded == nf, (decoded, nf)
    assert abs(output_fps-fps) < 1e-6
    partial.replace(DEST)
    scripts = ('render_full14.py', 'topology_voronoi.py', 'bvh_field.py', 'overlap_chords.py', 'prepare_full14.py')
    summary = dict(source=str(SOURCE), output=str(DEST), frames=decoded, width=w, height=h,
        fps=output_fps, duration_seconds=decoded/output_fps, full_decode_verified=True,
        contour_vertices=64, grid_scale=.5, raster_scale=1, network_thickness_pixels=2,
        frame_workers=args.workers, bvh_threads_per_worker=4,
        overlap_rule='open continuous polyline through ALL contour-ordered intersections',
        render_seconds=render_seconds, mean_compute_ms=float(np.mean(timings)),
        median_compute_ms=float(np.median(timings)), p95_compute_ms=float(np.percentile(timings, 95)),
        totals=dict(totals), overlap_issue_types=dict(issues), output_bytes=DEST.stat().st_size,
        script_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in scripts},
        limitations=['Grid topology is numerical, not exact analytic segment Voronoi.',
          'Failed junctions and unbracketed intervals are omitted and logged.',
          'No overlap divider is invented when the agreed intersection rule is undefined or ambiguous.'])
    (OUT / 'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
