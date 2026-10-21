"""Strict-compatible accelerations of the supplied video Voronoi implementation.

The polygon coordinates, grid, float precision, tie breaking, bisection count,
and OpenCV line rasterizer are unchanged. No contour simplification is done here.
"""
from pathlib import Path
import sys
import types
import inspect
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'deps'))
from numba import njit
import baseline_video as baseline

# Independent namespace: baseline remains available for honest comparisons.
_impl = types.ModuleType('_fast_video_impl')
_impl.__file__ = str(HERE / 'baseline_video.py')
exec(compile((HERE / 'baseline_video.py').read_text(encoding='utf-8'),
             _impl.__file__, 'exec'), _impl.__dict__)


@njit(cache=True, fastmath=False, nogil=True)
def _finish_distance(dot_a, dot_v, pnorm, aa, av, vv, safe):
    for i in range(dot_a.shape[0]):
        for j in range(dot_a.shape[1]):
            d = dot_a[i, j] * -2.0
            d += pnorm[i]
            d += aa[j]
            wv = dot_v[i, j] - av[j]
            t = min(1.0, max(0.0, wv / safe[j]))
            tmp = t * vv[j]
            tmp -= 2.0 * wv
            tmp *= t
            d += tmp
            dot_a[i, j] = math.sqrt(max(d, 0.0))
    return dot_a


def _dist_to_edges(Pc, aT, vT, vv, safe, aa, av, e_idx=None):
    if e_idx is not None:
        aT, vT = aT[:, e_idx], vT[:, e_idx]
        vv, safe, aa, av = vv[e_idx], safe[e_idx], aa[e_idx], av[e_idx]
    # Preserve the exact BLAS operations, including their rounding behavior.
    return _finish_distance(Pc @ aT, Pc @ vT, (Pc * Pc).sum(1), aa, av, vv, safe)


_impl._dist_to_edges = _dist_to_edges
build_field = _impl.build_field


@njit(cache=True, fastmath=False, nogil=True)
def _reduce_tile(dot_a, dot_v, pnorm, aa, av, vv, safe, candidates,
                 ids, K, prim, Dmn, D_ag, round_for_argmin=False):
    for i in range(len(ids)):
        best = np.inf
        best2 = np.inf
        bestj = -1
        pos = ids[i]
        last_ag = -1
        agent_best2 = np.inf
        for j in range(len(candidates)):
            d = dot_a[i, j] * -2.0
            d += pnorm[i]
            d += aa[j]
            wv = dot_v[i, j] - av[j]
            t = min(1.0, max(0.0, wv / safe[j]))
            tmp = t * vv[j]
            tmp -= 2.0 * wv
            tmp *= t
            d += tmp
            d = max(d, 0.0)
            edge = candidates[j]
            ag = edge // K
            if ag != last_ag:
                if last_ag >= 0:
                    D_ag[last_ag,pos] = np.float32(math.sqrt(agent_best2))
                last_ag = ag
                agent_best2 = d
            else:
                agent_best2 = min(agent_best2,d)
            if d < best2:
                # sqrt is monotone but can round distinct d2 to equal values.
                # Compare the rounded root to preserve the first-index tie rule.
                dist = math.sqrt(d)
                if round_for_argmin:
                    dist = float(np.float32(dist))
                if dist < best:
                    best = dist
                    bestj = edge
                best2 = d
        if last_ag >= 0:
            D_ag[last_ag,pos] = np.float32(math.sqrt(agent_best2))
        prim[pos] = bestj
        Dmn[pos] = np.float32(best)


FIELD_WORKERS = 1
_pools = {}


def set_field_workers(workers):
    global FIELD_WORKERS
    FIELD_WORKERS = max(1, int(workers))


def _field_pruned(Pf64, W, H, xs, ys, c0, aT, vT, vv, safe, aa, av,
                  e2a, M, K, tile=24):
    prim = np.full(W*H, -1, np.int32)
    Dmn = np.full(W*H, np.inf, np.float32)
    D_ag = np.full((M, W*H), np.inf, np.float32)
    jobs = [(ty,min(ty+tile,H),tx,min(tx+tile,W))
            for ty in range(0,H,tile) for tx in range(0,W,tile)]
    def run_jobs(batch):
        count = 0
        for ty,y1,tx,x1 in batch:
            cc = np.array([0.5*(xs[tx]+xs[x1-1])-c0[0],
                           0.5*(ys[ty]+ys[y1-1])-c0[1]])
            hd = 0.5*float(np.hypot(xs[x1-1]-xs[tx],ys[y1-1]-ys[ty]))
            w = cc[:,None]-aT
            t = (w*vT).sum(0)/safe
            np.clip(t,0.0,1.0,out=t)
            dv = w-t[None,:]*vT
            dd = np.sqrt((dv*dv).sum(0))
            lb = np.maximum(0.0,dd-hd)
            ei = np.nonzero(lb<=lb.min()+2.0*hd)[0]
            count += len(ei)
            ids = (np.arange(ty,y1)[:,None]*W+np.arange(tx,x1)[None,:]).ravel()
            pc = Pf64[ids]
            da, dv = pc@aT[:,ei], pc@vT[:,ei]
            _reduce_tile(da,dv,(pc*pc).sum(1),aa[ei],av[ei],vv[ei],safe[ei],
                         ei,ids,K,prim,Dmn,D_ag)
        return count
    if FIELD_WORKERS == 1:
        count = run_jobs(jobs)
    else:
        if FIELD_WORKERS not in _pools:
            _pools[FIELD_WORKERS] = ThreadPoolExecutor(max_workers=FIELD_WORKERS)
        count = sum(_pools[FIELD_WORKERS].map(run_jobs,
                    [jobs[i::FIELD_WORKERS] for i in range(FIELD_WORKERS)]))
    return prim,Dmn,D_ag,count/max(1,len(jobs)*len(e2a))


_impl._field_pruned = _field_pruned


def _field_full(Pf64, aT, vT, vv, safe, aa, av, M, K, rows_per_chunk):
    n = len(Pf64)
    prim = np.full(n,-1,np.int32)
    D = np.full(n,np.inf,np.float32)
    D_ag = np.full((M,n),np.inf,np.float32)
    ei = np.arange(M*K,dtype=np.int64)
    jobs = list(range(0,n,rows_per_chunk))
    def run_chunk(start):
        end = min(n,start+rows_per_chunk)
        pc = Pf64[start:end]
        _reduce_tile(pc@aT,pc@vT,(pc*pc).sum(1),aa,av,vv,safe,ei,
                     np.arange(start,end,dtype=np.int64),K,prim,D,D_ag,True)
    if FIELD_WORKERS == 1:
        for start in jobs:
            run_chunk(start)
    else:
        if FIELD_WORKERS not in _pools:
            _pools[FIELD_WORKERS] = ThreadPoolExecutor(max_workers=FIELD_WORKERS)
        list(_pools[FIELD_WORKERS].map(run_chunk,jobs))
    return prim,D,D_ag


_build_source = inspect.getsource(baseline.build_field)
_start = _build_source.index('        D_all = np.empty')
_end = _build_source.index('        keep_ratio = 1.0',_start)
_build_source = _build_source[:_start] + '''        prim,D,D_ag = _field_full(Pf64,aT,vT,vv,safe,aa,av,M,K,rows_per_chunk)
''' + _build_source[_end:]
_impl._field_full = _field_full
exec(compile(_build_source,str(HERE/'fast_build_generated.py'),'exec'),_impl.__dict__)
build_field = _impl.build_field


@njit(cache=True, fastmath=False, nogil=True)
def _bisect_compiled(p0, u, ea0, va, sa, eb0, vb, sb, g0):
    lo = np.zeros(len(p0), np.float64)
    hi = np.ones(len(p0), np.float64)
    for i in range(len(p0)):
        for _ in range(18):
            mid = 0.5*(lo[i]+hi[i])
            x = p0[i,0]+mid*u[i,0]
            y = p0[i,1]+mid*u[i,1]
            wx,wy = x-ea0[i,0],y-ea0[i,1]
            t = min(1.0,max(0.0,(wx*va[i,0]+wy*va[i,1])/sa[i]))
            dx,dy = wx-t*va[i,0],wy-t*va[i,1]
            da = math.sqrt(dx*dx+dy*dy)
            wx,wy = x-eb0[i,0],y-eb0[i,1]
            t = min(1.0,max(0.0,(wx*vb[i,0]+wy*vb[i,1])/sb[i]))
            dx,dy = wx-t*vb[i,0],wy-t*vb[i,1]
            db = math.sqrt(dx*dx+dy*dy)
            if (da-db)*g0[i] > 0:
                lo[i] = mid
            else:
                hi[i] = mid
    return lo,hi


def _bisect(p0,u,ea0,ea1,eb0,eb1,g0):
    va,vb = ea1-ea0,eb1-eb0
    sa,sb = (va*va).sum(1),(vb*vb).sum(1)
    sa,sb = np.where(sa>1e-12,sa,1.0),np.where(sb>1e-12,sb,1.0)
    return _bisect_compiled(p0,u,ea0,va,sa,eb0,vb,sb,g0)

# Reject same-agent transitions before distance evaluations. Their final keep
# mask is false in the original, so the order and values of survivors are exact.
_source = inspect.getsource(baseline.extract_network)
_source = _source.replace('ii, jj = np.nonzero(A != B)',
    'ii, jj = np.nonzero((A != B) if with_self else (e2a[A] != e2a[B]))')
_start = _source.index('        for _ in range(18):')
_end = _source.index('        s = 0.5 * (lo + hi)', _start)
_source = _source[:_start] + '        lo, hi = _bisect(p0,u,ea0,ea1,eb0,eb1,g0)\n' + _source[_end:]
_impl._bisect = _bisect
exec(compile(_source, str(HERE / 'fast_extract_generated.py'), 'exec'), _impl.__dict__)
extract_network = _impl.extract_network


@njit(cache=True, fastmath=False)
def _dedup(px, py, ded2):
    # Match NumPy float32 intermediates and the original greedy order exactly.
    n = len(px)
    keep = np.ones(n, np.bool_)
    threshold = np.float32(ded2)
    for a in range(n):
        if keep[a]:
            for b in range(n):
                if a != b:
                    dx, dy = px[b] - px[a], py[b] - py[a]
                    d = np.float32(np.float32(dx * dx) + np.float32(dy * dy))
                    if d <= threshold:
                        keep[b] = False
    return keep


@njit(cache=True, fastmath=False)
def _segments(ix, iy, tol2):
    # Compiled enumeration replaces Python bucket loops. Groups are individual
    # primitive pairs; no cross-group line can be created. Squared test is exact.
    n = len(ix)
    used = np.zeros(n, np.bool_)
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            dx, dy = int(ix[b]) - int(ix[a]), int(iy[b]) - int(iy[a])
            if dx * dx + dy * dy <= tol2:
                count += 1
    segs = np.empty((count, 2, 2), np.int32)
    j = 0
    for a in range(n):
        for b in range(a + 1, n):
            dx, dy = int(ix[b]) - int(ix[a]), int(iy[b]) - int(iy[a])
            if dx * dx + dy * dy <= tol2:
                segs[j, 0, 0], segs[j, 0, 1] = ix[a], iy[a]
                segs[j, 1, 0], segs[j, 1, 1] = ix[b], iy[b]
                j += 1
                used[a] = used[b] = True
    return segs, used


@njit(cache=True, fastmath=False, nogil=True)
def _raster_geometry(sgx, sgy, ends, ded2, tol2):
    segments = np.empty((max(16, len(sgx)*16), 2, 2), np.int32)
    circles = np.empty((len(sgx), 2), np.int32)
    ns = nc = 0
    for g in range(len(ends)-1):
        lo, hi = ends[g], ends[g+1]
        px, py = sgx[lo:hi], sgy[lo:hi]
        keep = _dedup(px, py, ded2)
        ix = np.round(px[keep]).astype(np.int32)
        iy = np.round(py[keep]).astype(np.int32)
        segs, used = _segments(ix, iy, tol2)
        if ns + len(segs) > len(segments):
            bigger = np.empty((max(len(segments)*2,ns+len(segs)),2,2),np.int32)
            bigger[:ns] = segments[:ns]
            segments = bigger
        segments[ns:ns+len(segs)] = segs
        ns += len(segs)
        for a in range(len(ix)):
            if not used[a]:
                circles[nc,0], circles[nc,1] = ix[a],iy[a]
                nc += 1
    return segments[:ns],circles[:nc]


def rasterize_network_hi(F, pts, codes, pkeys, ss=3, tol_field=2.4,
                         dedup_field=0.35, thick_img=2.0):
    h, w = F['H'] * ss, F['W'] * ss
    masks = [np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)]
    if len(pts) == 0:
        return tuple(masks)
    sc = float(F['scale']) * ss
    thick = max(1, int(round(float(thick_img) * sc)))
    cx = (pts[:, 0] - float(F['region'][0])) * sc
    cy = (pts[:, 1] - float(F['region'][1])) * sc
    ok = (cx >= -2) & (cx <= w + 2) & (cy >= -2) & (cy <= h + 2)
    cx, cy, codes, pkeys = cx[ok], cy[ok], codes[ok], pkeys[ok]
    for code, mask in zip((1, 2), masks):
        sel = codes == code
        gx, gy, pk = cx[sel], cy[sel], pkeys[sel]
        order = np.argsort(pk, kind='stable')
        spk = pk[order]
        sgx, sgy = gx[order], gy[order]
        ends = np.r_[0, np.flatnonzero(np.diff(spk)) + 1, len(spk)]
        segs,circles = _raster_geometry(sgx,sgy,ends,
            (float(dedup_field)*ss)**2,(float(tol_field)*ss)**2)
        if len(segs):
            cv2.polylines(mask, list(segs), False, 255, thick)
        for x,y in circles:
            cv2.circle(mask, (int(x),int(y)), max(1, thick//2), 255, -1)
    return tuple(masks)
