# -*- coding: utf-8 -*-
"""多边形站点的改版维诺图 —— 用于视频序列的版本。

与 `E:\\polygon-voronoi\\polygon_voronoi.py` 的差别
------------------------------------------------
1. **不要同速膨胀的动画过程**, 只要最终结果(分区 + 网络)。
2. 场景是**视频帧**, 每个 agent 的轮廓有 64 个采样点 —— 必须先**缩减成少数几条边**,
   否则 19 个对象 × 64 条边 = 1216 个图元, 图元对组合爆炸且大量是同一对象内部的噪声。
3. 视频特有的难点: **抽稀必须时间一致**。逐帧独立做 approxPolyDP 会得到
   数量与位置都抖动不定的角点 -> 维诺网络逐帧闪烁。
   做法: 固定顶点数 K + 在**框归一化坐标**下抽稀(消除尺度变化)
        + 循环位移对齐(让"第 i 个顶点"帧间对应) + 顶点级时间平滑。

几何内核完全沿用 polygon_voronoi.py 的结论:
    一条边 -> 一顶「帐篷」(沿边的平面脊 + 两端半圆锥)
    逐像素 argmin -> 图元标签图; 标签变化处 = 下包络折脊 = 网络
    二分线只有三种: 边-边 直线 / 边-顶点 抛物线 / 顶点-顶点 直线
    凸顶点「平局射线」需过滤(两图元最近点重合则丢弃)
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 1. 轮廓抽稀: 64 点 -> 固定 K 个顶点, 且时间一致
# ---------------------------------------------------------------------------
def _align_cycle(pts, ref):
    """在 ref 的 K 个顶点上找一个循环位移, 使 pts 与 ref 的对应误差最小。

    这是让"第 i 个顶点"帧间对应的关键: 角点本身没编号, 靠形状匹配对齐。
    """
    k = len(ref)
    best, bs = 1e18, 0
    for s in range(k):
        d = pts - np.roll(ref, s, axis=0)
        v = float((d * d).sum())
        if v < best:
            best, bs = v, s
    return np.roll(pts, -bs, axis=0)


def simplify_to_k(poly, k, box=None, eps0=0.012):
    """把轮廓抽稀成恰好 k 个顶点。

    返回 (k,2) float32。抽稀在**框归一化坐标**下做(消除尺度抖动),
    结果再还原到绝对像素坐标。

    不足 k 个角点时在最长边插入**中点**(共线点, 几何不变);
    超过 k 个角点时逐步提高 eps 重试, 仍超额则按「最短边优先合并」删到 k。
    """
    p = np.asarray(poly, np.float64)
    n = len(p)
    if n < 3:
        return None
    if box is None:
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        bw = max(float(np.ptp(p[:, 0])), 1.0)
        bh = max(float(np.ptp(p[:, 1])), 1.0)
    else:
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        bw = max(float(box[2] - box[0]), 1.0)
        bh = max(float(box[3] - box[1]), 1.0)

    def to_norm(q):
        return np.stack([(q[:, 0] - cx) / bw, (q[:, 1] - cy) / bh], 1)

    def to_abs(q):
        return np.stack([q[:, 0] * bw + cx, q[:, 1] * bh + cy], 1).astype(np.float32)

    q = to_norm(p)
    qc = np.ascontiguousarray(q.reshape(-1, 1, 2), np.float32)

    # --- 找角点
    ap = None
    for eps in (eps0, eps0 * 1.5, eps0 * 2.2, eps0 * 3.2, eps0 * 4.5, eps0 * 6.5):
        peri = cv2.arcLength(qc, True)
        a = cv2.approxPolyDP(qc, eps * peri, True).reshape(-1, 2).astype(np.float64)
        ap = a
        if len(a) <= k:
            break
    if ap is None or len(ap) < 3:
        return None

    # --- 角点太多: 按最短边优先合并, 删到 k
    pts = ap.copy()
    while len(pts) > k:
        m = len(pts)
        nxt = np.roll(pts, -1, axis=0)
        seg = np.linalg.norm(nxt - pts, axis=1)
        i = int(np.argmin(seg))                      # 最短边
        j = (i + 1) % m
        mid = 0.5 * (pts[i] + pts[j])                # 合并成中点
        pts = np.delete(pts, j, axis=0)
        pts[i] = mid

    # --- 角点不足: 在最长边插入中点(共线, 不改变几何)
    while len(pts) < k:
        m = len(pts)
        nxt = np.roll(pts, -1, axis=0)
        seg = np.linalg.norm(nxt - pts, axis=1)
        i = int(np.argmax(seg))
        mid = 0.5 * (pts[i] + nxt[i])
        pts = np.insert(pts, i + 1, mid, axis=0)

    return to_abs(pts)


def resample_uniform(poly, k):
    """按**弧长均匀**重采样闭合轮廓, 得到恰好 k 个点。

    为什么需要它(重要)
    ------------------
    `simplify_to_k` 是先用 approxPolyDP 抽「角点」, 再补共线中点凑到 k 个顶点。
    所以把 k 从 6 调到 64 **不会**增加任何形状细节 —— 多出来的点全落在直边上,
    多边形几何完全不变, 距离场和维诺网络一模一样, 只是白算 10 倍。
    本函数改为在弧长上等距取 k 个样本: k 越大越贴近原始轮廓, 这才是
    「尽可能使用较多采样点」的正确含义。

    相位(起点)取原始轮廓的第 0 个顶点; 帧间的相位漂移由 `simplify_stack` 的
    循环位移对齐 + 时间平滑吸收。k 较大时点距远小于需要分辨的形状尺度,
    残余相位误差对多边形形状的影响可忽略。
    """
    p = np.asarray(poly, np.float64)
    n = len(p)
    if n < 3 or k < 3:
        return None
    nxt = np.roll(p, -1, axis=0)
    seg = np.linalg.norm(nxt - p, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])          # (n+1,)
    L = float(cum[-1])
    if L < 1e-9:
        return None
    t = np.arange(k) * (L / k)
    i = np.searchsorted(cum, t, side="right") - 1
    np.clip(i, 0, n - 1, out=i)
    den = seg[i]
    den = np.where(den > 1e-12, den, 1.0)
    frac = (t - cum[i]) / den
    q = p[i] + frac[:, None] * (nxt[i] - p[i])
    return q.astype(np.float32)


def simplify_stack(polys, k, boxes=None, smooth_win=9, mode="corner"):
    """对**同一个 agent** 的整条时间序列做顶点缩减, 保证帧间顶点对应。

    polys : (T, 64, 2) 该 agent 逐帧的绝对坐标轮廓(可能含 None -> 由调用方剔除)
    返回   : (T, k, 2) float32

    步骤: 逐帧取 k 个顶点 -> 归一化 -> 相对第一帧循环位移对齐 -> 时间平滑 -> 还原

    mode
    ----
    "corner"  : approxPolyDP 抽角点(旧逻辑)。k 小(如 6)时是"六边形近似",
                但 k 调大不会增加细节(多出的点是共线中点)。
    "uniform" : 弧长均匀重采样。k 可以直接开到 32/64, 真正逼近原始轮廓。
    """
    T = len(polys)
    qs = np.empty((T, k, 2), np.float64)
    ok = np.zeros(T, bool)
    for t, p in enumerate(polys):
        if p is None:
            continue
        if mode == "uniform":
            r = resample_uniform(p, k)
        else:
            b = boxes[t] if boxes is not None else None
            r = simplify_to_k(p, k, b)
        if r is None:
            continue
        qs[t] = r
        ok[t] = 1
    if not ok.any():
        return None, ok

    # 归一化到"质心原点 + 平均半径单位"以便对齐与平滑
    def norm(q):
        c = q.mean(0)
        u = q - c
        s = float(np.linalg.norm(u, axis=1).mean())
        return u / (s if s > 1e-9 else 1.0), c, s

    idx = np.nonzero(ok)[0]
    # 用中位形状作为对齐参考
    refq, _, _ = norm(qs[idx[0]])
    for _ in range(2):
        stack = []
        for t in idx:
            u, _, _ = norm(qs[t])
            stack.append(_align_cycle(u, refq))
        stack = np.stack(stack)
        refq, _, _ = norm(stack.mean(0))

    # 对齐 + 去归一化回绝对坐标 -> 时间平滑
    absq = np.empty((len(idx), k, 2), np.float64)
    for m, t in enumerate(idx):
        u, c, s = norm(qs[t])
        absq[m] = _align_cycle(u, refq) * s + c
    if smooth_win >= 3 and len(idx) >= smooth_win:
        w = smooth_win if smooth_win % 2 == 1 else smooth_win + 1
        w = min(w, len(idx) if len(idx) % 2 == 1 else len(idx) - 1)
        if w >= 3:
            half = w // 2
            pad = np.pad(absq, ((half, half), (0, 0), (0, 0)), mode="edge")
            ker = np.ones(w) / w
            sm = np.empty_like(absq)
            for a in range(k):
                for b in range(2):
                    sm[:, a, b] = np.convolve(pad[:, a, b], ker, mode="valid")
            absq = sm

    out = np.empty((T, k, 2), np.float32)
    out[:] = np.nan
    out[idx] = absq.astype(np.float32)
    return out, ok


# ---------------------------------------------------------------------------
# 2. 场: 逐边「帐篷」距离 -> argmin -> 图元标签
# ---------------------------------------------------------------------------
def _dist_to_edges(Pc, aT, vT, vv, safe, aa, av, e_idx=None):
    """Pc (c,2) float64(已平移到区域中心) 到指定边集的距离, 返回 (c,ne) float32。

    e_idx=None 表示全部边。抽出成函数是为了让「全量」与「剪枝子集」共用同一套公式,
    保证两条路径数值完全一致。
    """
    if e_idx is None:
        a_, v_, vv_, sf_, aa_, av_ = aT, vT, vv, safe, aa, av
    else:
        a_ = aT[:, e_idx]
        v_ = vT[:, e_idx]
        vv_ = vv[e_idx]
        sf_ = safe[e_idx]
        aa_ = aa[e_idx]
        av_ = av[e_idx]
    A2 = (Pc * Pc).sum(1)[:, None]                    # (c,1)
    d2 = Pc @ a_                                      # P·a
    d2 *= -2.0
    d2 += A2
    d2 += aa_[None, :]                                # d2 = |P-a|^2
    wv = Pc @ v_
    wv -= av_[None, :]                                # w·v
    tc = wv / sf_[None, :]
    np.clip(tc, 0.0, 1.0, out=tc)
    # 唯一一条公式, 免去 where 分段:
    #   d^2 = |P-a|^2 - 2*tc*(w·v) + tc^2*|v|^2
    #   tc=0 -> |P-a|^2 (t<0);  tc=1 -> |P-b|^2 (t>1);  否则 到支撑线的距离
    tmp = tc * vv_[None, :]
    tmp -= 2.0 * wv
    tmp *= tc
    d2 += tmp
    np.maximum(d2, 0.0, out=d2)
    np.sqrt(d2, out=d2)
    return d2


def _field_pruned(Pf64, W, H, xs, ys, c0, aT, vT, vv, safe, aa, av,
                  e2a, M, K, tile=24):
    """分块剪枝建场: 每个 tile 只对「可能成为最近邻」的边算距离。

    剪枝是**精确的**, 不是近似 —— 推导:
        cc = tile 中心, hd = tile 半对角线(场像素)
        对任意边 e 与 tile 内任意像素 p:
            d_e(p) >= dist(cc, e) - hd  >=  max(0, dist(cc,e) - hd) =: LB_e
        而 d_{e*}(p) <= dist(cc, e*) + hd <= LB_{e*} + 2*hd    (e* 取 LB 最小者)
        故 LB_e > LB_min + 2*hd 的边, 在 tile 内任何像素上都不可能成为 argmin,
        可以安全剔除, 结果与全量计算**逐位相同**。
    被剔除 agent 的 D_ag 记 +inf: 它既然不可能是最近, 就永远赢不了分区 argmin。

    复杂度从 O(N*E) 降到 O(N*E_kept/tile大小), E_kept 通常只有 E 的 1/4 ~ 1/6。
    """
    E = e2a.shape[0]
    N = W * H
    prim = np.full(N, -1, np.int32)
    Dmn = np.full(N, np.inf, np.float32)
    D_ag = np.full((M, N), np.inf, np.float32)
    c0x, c0y = float(c0[0]), float(c0[1])
    n_keep = 0
    n_tile = 0

    for ty in range(0, H, tile):
        y1 = min(ty + tile, H)
        for tx in range(0, W, tile):
            x1 = min(tx + tile, W)
            cc = np.array([0.5 * (xs[tx] + xs[x1 - 1]) - c0x,
                           0.5 * (ys[ty] + ys[y1 - 1]) - c0y])
            hd = 0.5 * float(np.hypot(xs[x1 - 1] - xs[tx],
                                      ys[y1 - 1] - ys[ty]))
            # 中心到所有线段的距离 -> 下界
            w = cc[:, None] - aT                          # (2,E)
            t = (w * vT).sum(0) / safe
            np.clip(t, 0.0, 1.0, out=t)
            dv = w - t[None, :] * vT                      # (2,E)
            dd = np.sqrt((dv * dv).sum(0))                # (E,)
            LB = np.maximum(0.0, dd - hd)
            e_idx = np.nonzero(LB <= LB.min() + 2.0 * hd)[0]
            n_keep += len(e_idx)
            n_tile += 1

            rows = np.arange(ty, y1)
            cols = np.arange(tx, x1)
            idx = (rows[:, None] * W + cols[None, :]).ravel()
            d2 = _dist_to_edges(Pf64[idx], aT, vT, vv, safe, aa, av, e_idx)
            j = d2.argmin(1)
            prim[idx] = e_idx[j]
            Dmn[idx] = d2[np.arange(len(idx)), j]

            # 每个 agent 在该 tile 内的最小距离(边按 agent 排好序, e_idx 递增)
            aid = e_idx // K
            st = np.r_[0, np.flatnonzero(np.diff(aid)) + 1]
            mins = np.minimum.reduceat(d2, st, axis=1)
            ags = aid[st]
            for q in range(len(ags)):
                ag = int(ags[q])
                cur = D_ag[ag, idx]
                np.minimum(cur, mins[:, q], out=cur)
                D_ag[ag, idx] = cur

    ratio = float(n_keep) / max(1, n_tile * E)
    return prim, Dmn, D_ag, ratio


def build_field(polys, region, scale, rows_per_chunk=8192,
                prune=True, tile=24):
    """在 region=(x0,y0,x1,y1) 内以 1/scale 降采样建「帐篷」距离场。

    polys : (M, K, 2) 绝对像素坐标, 顶点数一致
    返回 dict(prim, D, D_ag, edges, e2a, ...)

    性能: 关键是写成**矩阵乘法**而不是逐图元广播。
    逐图元 Python 循环 0.9s/帧; (c,N,2) 显式广播 0.35~0.97s;
    矩阵乘法(P@v^T 等 3 个 (c,2)x(2,E) )只需约 30ms。

    距离公式(把 t 钳制分段处理, 免去构造 (c,N,2) 中间量):
        |P-a|^2 = |P|^2 - 2(P·a) + |a|^2
        (w·v)   = P·v - a·v
        t<0  -> |P-a|^2 ;  t>1 -> |P-b|^2 ;  否则 |P-a|^2 - (w·v)^2/|v|^2
    """
    x0, y0, x1, y1 = region
    W = max(2, int(round((x1 - x0) * scale)))
    H = max(2, int(round((y1 - y0) * scale)))
    xs = x0 + (np.arange(W) + 0.5) / scale
    ys = y0 + (np.arange(H) + 0.5) / scale
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y], -1)
    Pflat = np.ascontiguousarray(P.reshape(-1, 2), np.float32)
    N = Pflat.shape[0]

    M, K = len(polys), polys.shape[1]
    A = np.asarray(polys, np.float32)                     # (M,K,2)
    Bv = np.roll(A, -1, axis=1)
    edges = np.stack([A, Bv], 2).reshape(M * K, 2, 2)     # (M*K,2,2)
    e2a = np.repeat(np.arange(M), K)
    e2n = np.tile(np.arange(K), M)
    E = len(edges)

    # 精度: |P-a|^2 = |P|^2 - 2(P·a) + |a|^2 是**灾难性抵消**式,
    # 绝对坐标 ~2500 时 float32 只余 5 位有效数字 -> d 误差可达 1.0 px。
    # 所以: 先把坐标平移到区域中心(减小量级), 再用 float64 运算。
    # 平移到中心后 |P|^2 ~ 1.6e6, float64 的误差 ~1e-9, 完全够用。
    c0 = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], np.float64)
    Pf64 = Pflat.astype(np.float64) - c0

    a0 = np.ascontiguousarray(edges[:, 0].astype(np.float64)) - c0   # (E,2)
    b0 = np.ascontiguousarray(edges[:, 1].astype(np.float64)) - c0
    v = b0 - a0
    aT = np.ascontiguousarray(a0.T)                       # (2,E)
    vT = np.ascontiguousarray(v.T)
    vv = (v * v).sum(1)
    safe = np.where(vv > 1e-12, vv, 1.0)
    aa = (a0 * a0).sum(1)                                 # (E,)
    av = (a0 * v).sum(1)

    if not prune or tile <= 1 or E <= 256:
        # ---- 全量路径: 逐像素算所有边 ----
        D_all = np.empty((E, N), np.float32)
        for r0 in range(0, N, rows_per_chunk):
            r1 = min(N, r0 + rows_per_chunk)
            D_all[:, r0:r1] = _dist_to_edges(Pf64[r0:r1], aT, vT, vv, safe,
                                             aa, av).T
        prim = D_all.argmin(0).astype(np.int32)
        D = np.take_along_axis(D_all, prim[None], 0)[0]
        # 每个 agent 自身的距离场(取该 agent 所有边的最小) —— 供分区着色
        order = np.argsort(e2a, kind="stable")
        starts = np.searchsorted(e2a[order], np.arange(M))
        D_ag = np.minimum.reduceat(D_all[order], starts, axis=0)
        del D_all
        keep_ratio = 1.0
    else:
        # ---- 剪枝路径: 分块只算「可能成为最近邻」的边 ----
        # 结果与全量逐位相同(见 _field_pruned 的推导), 但快数倍。
        prim, D, D_ag, keep_ratio = _field_pruned(
            Pf64, W, H, xs, ys, c0, aT, vT, vv, safe, aa, av,
            e2a, M, K, tile=tile)

    Ag = e2a[prim]

    return dict(prim=prim.reshape(H, W), D=D.reshape(H, W), Ag=Ag.reshape(H, W),
                D_ag=D_ag.reshape(M, H, W), e2a=e2a, e2n=e2n, edges=edges,
                M=M, K=K, E=E, W=W, H=H, scale=scale, xs=xs, ys=ys,
                region=region, Pflat=Pflat, polys=A, X=X, Y=Y,
                keep_ratio=keep_ratio, tile=tile, prune=bool(prune))


def seg_dist_pairs(P, A, B):
    """**成对**的点到线段距离: P[i] 到线段 A[i]-B[i]。全部向量化。

    返回 (dist, t_raw)。t_raw 落在 (0,1) 内 = 垂足在线段内部(平面部分),
    被钳制到 0/1 = 走端点(圆锥部分) —— 这就是「帐篷」。
    """
    v = B - A
    w = P - A
    vv = (v * v).sum(1)
    vv_safe = np.where(vv > 1e-12, vv, 1.0)
    t = (w * v).sum(1) / vv_safe
    tc = np.clip(t, 0.0, 1.0)
    d = np.sqrt(((w - tc[:, None] * v) ** 2).sum(1))
    return d, t


def _nearest_on_seg(pt, a, b):
    v = b - a
    w = pt - a
    d2 = float(v @ v)
    if d2 < 1e-12:
        return a
    t = min(1.0, max(0.0, float(w @ v) / d2))
    return a + t * v


def extract_network(F, with_self=False, tie_tol=1e-4):
    """提取改版维诺网络。

    返回 (pts, codes, pairkeys, gids)
        pts      (N,2) float32  绝对像素坐标, 已亚像素细化
        codes    (N,)  int8     1=直线(边-边/顶点-顶点) 2=抛物线(边-顶点) 3=自中轴
        pairkeys (N,)  int64    图元对编码 min*(E+1)+max, 供调用方按弧串联
        gids     (N,)  int8     采样对编号 0..3

    关于 gids (重要)
    ----------------
    4 个像素偏移对各自独立扫描一遍边界, 于是同一条弧会得到 **4 条互相错开约
    1 个场像素的平行点列**。它们是同一根曲线的 4 份采样, 必须**分别串联**;
    若混在一起连成一条链, 就会在 4 条轨迹之间来回跳, 视觉上就是锯齿/串珠。
    """
    prim, edges, e2a = F["prim"], F["edges"], F["e2a"]
    H, W, E = F["H"], F["W"], F["E"]
    M = int(F["M"])
    Pflat = F["Pflat"]

    PX, PY, IA, IB, GC, GG, A2 = [], [], [], [], [], [], []
    # 4 个像素偏移对, 合起来覆盖全 8 邻域
    for gi, ((i0, j0), (i1, j1)) in enumerate(
            [((0, 0), (0, 1)), ((0, 0), (1, 0)),
             ((0, 0), (1, 1)), ((1, 0), (0, 1))]):
        hh = H - max(i0, i1)
        ww = W - max(j0, j1)
        if hh <= 0 or ww <= 0:
            continue
        A = prim[i0:i0 + hh, j0:j0 + ww]
        B = prim[i1:i1 + hh, j1:j1 + ww]
        ii, jj = np.nonzero(A != B)
        if len(ii) == 0:
            continue
        ia = A[ii, jj].astype(np.int64)
        ib = B[ii, jj].astype(np.int64)
        p0 = Pflat[(ii + i0) * W + (jj + j0)]      # 两端的栅格点坐标
        p1 = Pflat[(ii + i1) * W + (jj + j1)]

        # 沿 p0->p1 的 (d_a - d_b) 过零点: g0/g1 异号, 下面二分求精确根
        da0, _ = seg_dist_pairs(p0, edges[ia, 0], edges[ia, 1])
        db0, _ = seg_dist_pairs(p0, edges[ib, 0], edges[ib, 1])
        da1, _ = seg_dist_pairs(p1, edges[ia, 0], edges[ia, 1])
        db1, _ = seg_dist_pairs(p1, edges[ib, 0], edges[ib, 1])
        g0 = da0 - db0
        g1 = da1 - db1

        # ---- 亚像素 refinement: 在**真实距离函数**上二分求 d_a = d_b ----
        # 线性插值只在网格边上按比例估位置, 误差可达半个场像素, 于是相邻交点
        # 落在真实曲线两侧交替 -> 折线呈"N/M 形台阶"。改成在 p0->p1 上对
        #   g(s) = d_a(p0+s u) - d_b(p0+s u),  u = p1 - p0
        # 做二分, 得到的点严格落在真曲线上, 且与场分辨率无关。
        u = (p1 - p0).astype(np.float64)
        lo = np.zeros(len(u), np.float64)
        hi = np.ones(len(u), np.float64)
        ea0, ea1 = edges[ia, 0], edges[ia, 1]
        eb0, eb1 = edges[ib, 0], edges[ib, 1]
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            pm = p0 + mid[:, None] * u
            gam, _ = seg_dist_pairs(pm, ea0, ea1)
            gbm, _ = seg_dist_pairs(pm, eb0, eb1)
            gm = gam - gbm
            same = (gm * g0) > 0          # 与 lo 端同号 -> 根在右半
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        s = 0.5 * (lo + hi)
        px = p0[:, 0] + s * u[:, 0]
        py = p0[:, 1] + s * u[:, 1]
        pt = np.stack([px, py], 1)

        # 平局过滤: 凸顶点外法向锥内两相邻边给出完全相同的最近点 -> 伪边界
        na = nearest_pairs(pt, edges[ia, 0], edges[ia, 1])
        nb = nearest_pairs(pt, edges[ib, 0], edges[ib, 1])
        notie = np.linalg.norm(na - nb, axis=1) >= tie_tol

        ta = t_raw_pairs(pt, edges[ia, 0], edges[ia, 1])
        tb = t_raw_pairs(pt, edges[ib, 0], edges[ib, 1])
        va = (ta <= 1e-3) | (ta >= 1 - 1e-3)
        vb = (tb <= 1e-3) | (tb >= 1 - 1e-3)
        is_self = e2a[ia] == e2a[ib]
        # 自中轴(同一 agent 内部两图元之间)默认不要: 那是对象自己的骨架
        keep = notie & (with_self | (~is_self))
        if not keep.any():
            continue
        g = np.where(is_self, np.int8(3),
                     np.where(va == vb, np.int8(1), np.int8(2)))
        PX.append(px[keep]); PY.append(py[keep])
        IA.append(ia[keep]); IB.append(ib[keep]); GC.append(g[keep])
        GG.append(np.full(int(keep.sum()), gi, np.int8))
        # 智能体对编码: 两个对象之间的边界是一条连通曲线, 用它分组最稳
        aa = e2a[ia[keep]].astype(np.int64)
        bb = e2a[ib[keep]].astype(np.int64)
        A2.append(np.minimum(aa, bb) * (M + 1) + np.maximum(aa, bb))

    if not PX:
        z = np.zeros(0, np.float32)
        z8 = np.zeros(0, np.int8)
        zi = z.astype(np.int64)
        return (np.zeros((0, 2), np.float32), z8, zi, z8, zi)
    px = np.concatenate(PX); py = np.concatenate(PY)
    ia = np.concatenate(IA); ib = np.concatenate(IB)
    code = np.concatenate(GC)
    gid = np.concatenate(GG)
    a2key = np.concatenate(A2)
    lo = np.minimum(ia, ib)
    hi = np.maximum(ia, ib)
    return (np.stack([px, py], 1).astype(np.float32), code,
            lo * (E + 1) + hi, gid, a2key)


def nearest_pairs(P, A, B):
    """**成对**最近点: P[i] 在线段 A[i]-B[i] 上的投影点。"""
    v = B - A
    w = P - A
    d2 = (v * v).sum(1)
    safe = np.where(d2 > 1e-12, d2, 1.0)
    t = np.clip((w * v).sum(1) / safe, 0.0, 1.0)
    return A + t[:, None] * v


def t_raw_pairs(P, A, B):
    """**成对**未钳制投影参数 t: <0 或 >1 说明最近点在端点(走圆锥部分)。"""
    v = B - A
    w = P - A
    d2 = (v * v).sum(1)
    safe = np.where(d2 > 1e-12, d2, 1.0)
    return (w * v).sum(1) / safe


# ---------------------------------------------------------------------------
# 4. 网络栅格化: 在场分辨率里"连点成线", 再整体放大
# ---------------------------------------------------------------------------
def rasterize_network_field(F, pts, codes, pkeys, dilate=3):
    """把网络点栅格化成**连续**的线掩膜(在场分辨率下)。

    为什么必须"连点成线": 网格单元边界最多只产生 1 个交点, 沿曲线相邻交点
    间距恰好 = 1 个场像素; 逐点打像素会得到互不相邻的点阵(虚线/点阵)。
    做法同 polygon-voronoi 的 HTML 修复: 按图元对分组 -> 组内最近邻连线 ->
    场分辨率下 1px 粗的线 -> 调用方放大回原分辨率。

    返回 (mask1, mask2) 两张 uint8 掩膜(场分辨率), 分别对应 直线/抛物线。
    """
    h, w = F["H"], F["W"]
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    if len(pts) == 0:
        return m1, m2
    x0, y0 = F["region"][0], F["region"][1]
    sc = F["scale"]
    fx = (pts[:, 0] - x0) * sc - 0.5
    fy = (pts[:, 1] - y0) * sc - 0.5
    ok = (fx >= -1) & (fx <= w) & (fy >= -1) & (fy <= h)
    if not ok.any():
        return m1, m2
    fx, fy, codes, pkeys = fx[ok], fy[ok], codes[ok], pkeys[ok]

    TOL = 2.2                      # 场像素; 相邻交点间距 = 1, 留一点余量
    cell = int(np.ceil(TOL))
    for code, mask in ((1, m1), (2, m2)):
        sel = codes == code
        if not sel.any():
            continue
        ix = np.round(fx[sel]).astype(np.int32)
        iy = np.round(fy[sel]).astype(np.int32)
        pk = pkeys[sel]
        # 按图元对分组: 同组的点才属同一条弧, 不同弧绝不能互连
        order = np.argsort(pk, kind="stable")
        spk = pk[order]
        bnd = np.flatnonzero(np.diff(spk)) + 1
        for s in np.split(order, bnd):
            n = len(s)
            if n == 0:
                continue
            if n == 1:
                mask[iy[s[0]], ix[s[0]]] = 255
                continue
            # s 是全局下标; 下面一律用 0..n-1 的**局部**下标访问 xi/yi
            xi = ix[s]
            yi = iy[s]
            cxd = xi // cell
            cyd = yi // cell
            buckets = {}
            for m in range(n):
                buckets.setdefault((cxd[m], cyd[m]), []).append(m)
            used = np.zeros(n, bool)
            for m in range(n):
                cx, cy = cxd[m], cyd[m]
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        arr = buckets.get((cx + ox, cy + oy))
                        if not arr:
                            continue
                        for q in arr:
                            if q <= m:
                                continue
                            ddx = int(xi[q]) - int(xi[m])
                            ddy = int(yi[q]) - int(yi[m])
                            if ddx * ddx + ddy * ddy <= TOL * TOL:
                                cv2.line(mask, (int(xi[m]), int(yi[m])),
                                         (int(xi[q]), int(yi[q])), 255, 1)
                                used[m] = used[q] = True
            for m in range(n):
                if not used[m]:
                    mask[int(yi[m]), int(xi[m])] = 255
    if dilate > 1:
        k = np.ones((dilate, dilate), np.uint8)
        m1 = cv2.dilate(m1, k)
        m2 = cv2.dilate(m2, k)
    return m1, m2


def rasterize_network_ss(F, pts, codes, pkeys, ss=2, tol_field=2.2, thick_img=2):
    """超采样栅格化: 在 (场分辨率 × ss) 的网格上连点成线, 返回两张**软边**掩膜。

    为什么不用"直接把点连到原分辨率": 4 个 offset 采样对会让同一条弧产生 4 套
    互相错开 ~0.5 场像素的点, 直接最近邻连线会把它们连成来回折的锯齿带; 而按
    场分辨率栅格化再最近邻放大会得到 1 场像素 = 4 屏幕像素的方块。

    折中方案: 在 ss 倍超采样网格上连线(保留亚像素位置), 再由调用方用
    INTER_LINEAR 放大回原图 —— 既有已验证的连通性, 又是抗锯齿的细线。

    参数
    ----
    ss          : 超采样倍数(2 已足够)
    tol_field   : 相邻交点间距容差, 单位=场像素(1 场像素 = 1/scale 原图像素)
    thick_img   : 期望的线宽(原图像素), 会换算到超采样网格上

    返回 (m1, m2): uint8 掩膜, 尺寸 (H*ss, W*ss), 255=线。
    """
    h, w = F["H"] * ss, F["W"] * ss
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    if len(pts) == 0:
        return m1, m2
    x0, y0 = float(F["region"][0]), float(F["region"][1])
    sc = float(F["scale"])
    fx = (pts[:, 0] - x0) * sc * ss - 0.5
    fy = (pts[:, 1] - y0) * sc * ss - 0.5
    ok = (fx >= -1) & (fx <= w) & (fy >= -1) & (fy <= h)
    if not ok.any():
        return m1, m2
    fx, fy, codes, pkeys = fx[ok], fy[ok], codes[ok], pkeys[ok]

    tol = float(tol_field) * ss
    cell = int(np.ceil(tol))
    # 聚类去重: 4 个 offset 采样对会对同一条弧给出 4 套互相错开 ~0.5 场像素的点。
    # 若直接两两连线, 曲线会在这些副本之间来回折 -> 串珠/锯齿。先按 1 场像素
    # 分箱取质心(即亚像素细化), 每个箱只留一个点, 再连成链。
    cbin = max(1, int(round(ss)))
    th = max(1, int(round(float(thick_img) * sc * ss)))
    for code, mask in ((1, m1), (2, m2)):
        sel = codes == code
        if not sel.any():
            continue
        pk = pkeys[sel]
        order = np.argsort(pk, kind="stable")
        spk = pk[order]
        bnd = np.flatnonzero(np.diff(spk)) + 1
        gx, gy = fx[sel][order], fy[sel][order]
        for s in np.split(np.arange(len(spk)), bnd):
            if len(s) == 0:
                continue
            px, py = gx[s], gy[s]
            # 分箱 -> 质心
            bx = np.floor(px / cbin).astype(np.int64)
            by = np.floor(py / cbin).astype(np.int64)
            key = bx * 1000003 + by
            uk, inv = np.unique(key, return_inverse=True)
            ncl = len(uk)
            if ncl <= 1:
                cx = np.array([px.mean()])
                cy = np.array([py.mean()])
            else:
                cnt = np.bincount(inv, minlength=ncl).astype(np.float64)
                cx = np.bincount(inv, weights=px, minlength=ncl) / cnt
                cy = np.bincount(inv, weights=py, minlength=ncl) / cnt
            ix = np.round(cx).astype(np.int32)
            iy = np.round(cy).astype(np.int32)
            n = ncl
            if n == 1:
                cv2.circle(mask, (int(ix[0]), int(iy[0])), max(1, th // 2), 255, -1)
                continue
            cxd = ix // cell
            cyd = iy // cell
            buckets = {}
            for m in range(n):
                buckets.setdefault((cxd[m], cyd[m]), []).append(m)
            used = np.zeros(n, bool)
            for m in range(n):
                cxm, cym = cxd[m], cyd[m]
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        arr = buckets.get((cxm + ox, cym + oy))
                        if not arr:
                            continue
                        for q in arr:
                            if q <= m:
                                continue
                            ddx = int(ix[q]) - int(ix[m])
                            ddy = int(iy[q]) - int(iy[m])
                            if ddx * ddx + ddy * ddy <= tol * tol:
                                cv2.line(mask, (int(ix[m]), int(iy[m])),
                                         (int(ix[q]), int(iy[q])), 255, th)
                                used[m] = used[q] = True
            for m in range(n):
                if not used[m]:
                    cv2.circle(mask, (int(ix[m]), int(iy[m])),
                               max(1, th // 2), 255, -1)
    return m1, m2


def _mst_paths(P, max_edge=3.2):
    """把"沿曲线采样"的点重排成若干条**有序折线**, 返回下标数组列表。

    为什么不用"贪心最近邻游走"
    --------------------------
    交点间距 ≈ 1 场像素, 而邻接容差必须略大才能容忍疏密不均; 于是每个点会连上
    3~5 个邻居, 图里**没有度为 1 的端点**, 贪心游走从中途起步走几步就撞到已访问
    点而停止 —— 一条完整的弧被切成许多短段, 画出来就是虚线。

    为什么还要"剪长边"
    ------------------
    同一个图元对(两条边的平分线)可能被**别的对象切成好几段互不相连的曲线**,
    它们共用一个 pkey。MST 会用一条很长的边把这些独立段串起来, 直径路径就跨越
    了空隙 -> 图上出现横跨的长直线。因此建好 MST 后剪掉长度 > `max_edge` 的边,
    得到森林; 每棵树**各自**取直径路径(两次 BFS 即得最长链)。

    P        : (n,2) float64 点坐标(建议用场坐标, 数值条件更好)
    max_edge : MST 边的长度上限(场像素)。曲线内相邻点间距 ≈1, 取 3 已很宽松。

    返回 [idx (ni,) int64, ...], 每个元素是一条有序折线; 孤立点单独成一条。
    """
    n = len(P)
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], np.int64)]
    INF = np.inf
    in_tree = np.zeros(n, bool)
    key = np.full(n, INF)
    par = np.full(n, -1, np.int64)
    key[0] = 0.0
    mst = [[] for _ in range(n)]        # MST 邻接(带边长)
    for _ in range(n):
        u = int(np.argmin(np.where(in_tree, INF, key)))
        in_tree[u] = True
        if par[u] >= 0:
            w = float(np.sqrt(key[u]))
            mst[u].append((int(par[u]), w))
            mst[int(par[u])].append((u, w))
        du = ((P - P[u]) ** 2).sum(1)
        upd = (~in_tree) & (du < key)
        key[upd] = du[upd]
        par[upd] = u

    # 剪长边 -> 森林
    adj = [[v for (v, w) in mst[i] if w <= max_edge] for i in range(n)]

    def _bfs(s, comp):
        dist = {s: 0}
        prev = {s: -1}
        stack = [s]
        far = s
        while stack:
            c = stack.pop()
            for w in adj[c]:
                if w not in dist:
                    dist[w] = dist[c] + 1
                    prev[w] = c
                    if dist[w] > dist[far]:
                        far = w
                    stack.append(w)
        return far, prev, set(dist)

    seen = np.zeros(n, bool)
    out = []
    for s0 in range(n):
        if seen[s0]:
            continue
        a, _, members = _bfs(s0, None)
        for m in members:
            seen[m] = True
        if len(members) == 1:
            out.append(np.array([a], np.int64))
            continue
        b, prev, _ = _bfs(a, None)
        path = []
        c = b
        while c >= 0:
            path.append(c)
            c = prev[c]
        out.append(np.array(path[::-1], np.int64))
    return out


def _order_curve(P, r_n=1.8, r_bridge=2.8):
    """把"沿曲线采样"的点排成若干条有序链, **保证覆盖所有点**。

    为什么不用 MST 直径路径
    ----------------------
    MST 的直径路径只取最长的一条, 落在分支上的点会被**丢掉** —— 曲线稍有分叉
    (或采样疏密不均导致 MST 长出小枝)就会在图上留下缺口。

    这里的做法: 从"邻居最少的点"起, 反复走向**最近的未访问点**; 若最近的未访问点
    距离 ≤ `r_n` 视为沿曲线前进, ≤ `r_bridge` 视为"缺了一个采样点"直接跨过去,
    更远则说明是另一段曲线, 结束当前链、另起一条。这样每个点都进入某条链, 且链内
    相邻点间距受 `r_bridge` 约束(线性插值补上缺的点即可, 视觉上无缺口)。

    r_n      : 沿曲线前进的步长上限(场像素)。曲线内相邻点间距 ≈0.6~1.0。
    r_bridge : 允许跨越的缺口上限(场像素)。再大就认为是不同的曲线段。

    返回 [idx (ni,) int64, ...]。
    """
    n = len(P)
    if n == 0:
        return []
    if n <= 2:
        return [np.arange(n, dtype=np.int64)]
    D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D2, np.inf)
    rn2 = float(r_n) ** 2
    rb2 = float(r_bridge) ** 2
    deg = (D2 <= rn2).sum(1)
    unv = np.ones(n, bool)
    start = int(np.argmin(deg))
    chains = []
    while unv.any():
        cur = start if unv[start] else int(np.flatnonzero(unv)[0])
        chain = [cur]
        unv[cur] = False
        while True:
            d = D2[cur].copy()
            d[~unv] = np.inf
            j = int(np.argmin(d))
            dj = d[j]
            if not np.isfinite(dj) or dj > rb2:
                break
            cur = j
            chain.append(cur)
            unv[cur] = False
        chains.append(np.array(chain, np.int64))
    return chains


def rasterize_network_hi(F, pts, codes, pkeys, ss=3, tol_field=2.4,
                         dedup_field=0.35, thick_img=2.0):
    """在高分辨率画布上栅格化网络, 返回 (m1, m2) 两张掩膜。

    为什么这样做
    ------------
    * 场分辨率栅格化(`rasterize_network_field`)的连线逻辑是稳健的: 同一图元对内,
      只要两点距离在容差内就连线, **不依赖点的先后顺序**, 因此不会出现缺口。
      但它画在 (H,W) 的场网格上, 1 场像素 = 1/scale 原图像素, 放大后是方块。
    * 折线路径(`network_polylines`)与分辨率无关, 但依赖"把弧上的点排成正确顺序",
      而维诺边界会被别的对象打断、采样疏密不均, 排序难免漏点 -> 小缺口。

    这里取两者的长处: **沿用场掩膜的连线逻辑, 但把画布放大 ss 倍**。交点本身是
    亚像素精度(在真实距离函数上二分求得), 所以放大后线条位置依然准确; ss=3 时
    1 场像素 = 3 个子像素, 再由调用方缩小 -> 抗锯齿细线, 既连续又不块状。

    参数
    ----
    ss          : 画布相对场分辨率的放大倍数
    tol_field   : 连线距离容差(场像素)。曲线内相邻点间距 ≈0.6~1.0, 取 2.4 可跨过
                  采样稀疏处; 因已按图元对分组, 不会跨弧误连。
    dedup_field : 去重半径(场像素), 合并 4 个采样对给出的近重合点。
    thick_img   : 线宽(**原图像素**), 内部换算成画布子像素单位。

    返回 (m1, m2): uint8, 尺寸 (H*ss, W*ss), 255 = 线。
    """
    h, w = F["H"] * ss, F["W"] * ss
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    if len(pts) == 0:
        return m1, m2
    x0, y0 = float(F["region"][0]), float(F["region"][1])
    sc = float(F["scale"]) * ss                      # 原图 -> 画布
    thick = max(1, int(round(float(thick_img) * sc)))
    cx = (pts[:, 0] - x0) * sc
    cy = (pts[:, 1] - y0) * sc
    ok = (cx >= -2) & (cx <= w + 2) & (cy >= -2) & (cy <= h + 2)
    if not ok.any():
        return m1, m2
    cx, cy, codes, pkeys = cx[ok], cy[ok], codes[ok], pkeys[ok]

    tol = float(tol_field) * ss
    ded = float(dedup_field) * ss
    cell = max(1, int(np.ceil(tol)))
    for code, mask in ((1, m1), (2, m2)):
        sel = codes == code
        if not sel.any():
            continue
        gx, gy = cx[sel], cy[sel]
        pk = pkeys[sel]
        order = np.argsort(pk, kind="stable")
        spk = pk[order]
        bnd = np.flatnonzero(np.diff(spk)) + 1
        sgx, sgy = gx[order], gy[order]
        for s in np.split(np.arange(len(spk)), bnd):
            n = len(s)
            if n == 0:
                continue
            if n == 1:
                cv2.circle(mask, (int(round(sgx[s[0]])), int(round(sgy[s[0]]))),
                           max(1, thick // 2), 255, -1)
                continue
            px, py = sgx[s], sgy[s]
            # 去重: 4 个采样对高度重合, 合并后只保留一个代表点
            keep = np.ones(n, bool)
            for a in range(n):
                if not keep[a]:
                    continue
                d2 = (px - px[a]) ** 2 + (py - py[a]) ** 2
                dup = d2 <= ded * ded
                dup[a] = False
                keep[dup] = False
            px, py = px[keep], py[keep]
            m = len(px)
            if m < 1:
                continue
            ix = np.round(px).astype(np.int32)
            iy = np.round(py).astype(np.int32)
            if m == 1:
                cv2.circle(mask, (int(ix[0]), int(iy[0])),
                           max(1, thick // 2), 255, -1)
                continue
            # 网格分桶 + 邻域两两连线(不依赖顺序)
            cxd = ix // cell
            cyd = iy // cell
            buckets = {}
            for a in range(m):
                buckets.setdefault((cxd[a], cyd[a]), []).append(a)
            used = np.zeros(m, bool)
            for a in range(m):
                cxa, cya = cxd[a], cyd[a]
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        arr = buckets.get((cxa + ox, cya + oy))
                        if not arr:
                            continue
                        for b in arr:
                            if b <= a:
                                continue
                            ddx = int(ix[b]) - int(ix[a])
                            ddy = int(iy[b]) - int(iy[a])
                            if ddx * ddx + ddy * ddy <= tol * tol:
                                cv2.line(mask, (int(ix[a]), int(iy[a])),
                                         (int(ix[b]), int(iy[b])), 255, thick)
                                used[a] = used[b] = True
            for a in range(m):
                if not used[a]:
                    cv2.circle(mask, (int(ix[a]), int(iy[a])),
                               max(1, thick // 2), 255, -1)
    return m1, m2


def network_polylines(pts, codes, pkeys, gids, scale, a2keys=None,
                      dedup_field=0.35, min_pts=2, use_pairs=(0, 1, 2, 3),
                      r_n=1.8, r_bridge=2.8, bridge=True):
    """把网络点整理成**有序折线**, 返回 [(折线 (Ni,2) float32, 每点 code (Ni,) int8)]。

    为什么按**智能体对**分组, 而不是按图元对
    ----------------------------------------
    "两个对象之间的边界"是一条**连通曲线**; 但按"图元对(边的组合)"分组时, 同一条
    曲线会在**直线段与抛物线段的交界处换组** —— 于是曲线被切成两段, 画出来正好在
    交界处留一个小缺口。改用 `a2keys`(智能体对)分组后整条边界始终在同一组内;
    颜色不再由分组决定, 而是**逐点携带** code, 绘制时切成若干同色段(共享交界点)。

    组内流程
    --------
    1. 去重: 4 个采样对高度重合(最近邻中位仅 0.17 场像素), 半径 `dedup_field` 合并;
    2. `_order_curve` 重建曲线点序并覆盖所有点;
    3. 相邻点间距 > `r_n` 时线性插值补点(`bridge=True`), 消除稀疏处的小缺口。

    `a2keys` 为 None 时退回按图元对分组(向后兼容)。
    """
    if len(pts) == 0:
        return []
    sc = float(scale)
    fx = pts[:, 0] * sc
    fy = pts[:, 1] * sc
    out = []
    r2 = dedup_field * dedup_field
    keep_pair = np.isin(gids, np.asarray(use_pairs, np.int8))
    group_key = pkeys if a2keys is None else a2keys

    sel0 = keep_pair & (codes != 3)
    if not sel0.any():
        return []
    gsel = np.flatnonzero(sel0)
    order = np.argsort(group_key[gsel], kind="stable")
    gsel = gsel[order]
    gk = group_key[gsel]
    bnd = np.flatnonzero(np.diff(gk)) + 1
    for s in np.split(gsel, bnd):
        n = len(s)
        if n == 0:
            continue
        if n == 1:
            out.append((pts[s][:, :2].astype(np.float32),
                        codes[s].astype(np.int8)))
            continue
        px, py = fx[s], fy[s]
        # --- 1) 去重 ---
        keep = np.ones(n, bool)
        for a in range(n):
            if not keep[a]:
                continue
            d2 = (px - px[a]) ** 2 + (py - py[a]) ** 2
            dup = d2 <= r2
            dup[a] = False
            keep[dup] = False
        idx = s[keep]
        m = len(idx)
        if m < min_pts:
            continue
        Pc = np.stack([fx[idx], fy[idx]], 1).astype(np.float64)
        # --- 2) 排序成链 ---
        for seq in _order_curve(Pc, r_n=r_n, r_bridge=r_bridge):
            if len(seq) < min_pts:
                continue
            q = pts[idx[seq]][:, :2].astype(np.float32)
            cd = codes[idx[seq]].astype(np.int8)
            # --- 3) 桥接小缺口(在**原图坐标**上插值) ---
            if bridge and len(q) >= 2:
                d = np.sqrt(((q[1:] - q[:-1]) ** 2).sum(1))
                big = np.flatnonzero(d > r_n / sc)
                if len(big):
                    qs, cds = [q[0:1]], [cd[0:1]]
                    for i in range(1, len(q)):
                        if d[i - 1] > r_n / sc:
                            nins = min(4, int(np.ceil(d[i - 1] / (r_n / sc))) - 1)
                            if nins > 0:
                                t = np.linspace(0, 1, nins + 2)[1:-1, None]
                                mid = q[i - 1] * (1 - t) + q[i] * t
                                qs.append(mid.astype(np.float32))
                                cds.append(np.full(nins, cd[i - 1], np.int8))
                        qs.append(q[i:i + 1])
                        cds.append(cd[i:i + 1])
                    q = np.concatenate(qs, 0)
                    cd = np.concatenate(cds, 0)
            out.append((q, cd))
    return out


def draw_network_full(frame, pts, codes, pkeys, region, scale,
                      thick=2, col_line=(255, 255, 255), col_par=(0, 160, 230),
                      tol_field=2.2):
    """把网络点**直接连成线画在原分辨率帧上**(抗锯齿), 避免场分辨率放大的阶梯。

    与 `rasterize_network_field` 是同一套逻辑(按图元对分组 -> 组内最近邻连线),
    区别只在坐标系: 这里把场坐标还原成原图坐标, 用 `cv2.LINE_AA` 画粗线,
    所以曲线光滑、没有 1 场像素 = 4 屏幕像素的方块感。

    - `tol_field`: 相邻交点间距约定值(场像素), 换算到原图需除以 `scale`。
    - 连线严格限制在同一图元对内, 不同弧绝不互连。
    - 只有单独一个点、周围无可连对象时, 才退化成一个圆点(避免漏画)。
    """
    if len(pts) == 0:
        return 0, 0
    x0, y0 = float(region[0]), float(region[1])
    uk = 1.0 / max(scale, 1e-9)
    ux = (pts[:, 0] - x0) * uk
    uy = (pts[:, 1] - y0) * uk
    H, W = frame.shape[:2]
    ok = (ux >= -2) & (ux <= W + 2) & (uy >= -2) & (uy <= H + 2)
    if not ok.any():
        return 0, 0
    ux, uy, codes, pkeys = ux[ok], uy[ok], codes[ok], pkeys[ok]

    tol = float(tol_field) * uk
    cell = int(np.ceil(tol))
    nseg = 0
    for code, col in ((1, col_line), (2, col_par)):
        sel = codes == code
        if not sel.any():
            continue
        ix = np.round(ux[sel]).astype(np.int32)
        iy = np.round(uy[sel]).astype(np.int32)
        pk = pkeys[sel]
        order = np.argsort(pk, kind="stable")
        spk = pk[order]
        bnd = np.flatnonzero(np.diff(spk)) + 1
        for s in np.split(order, bnd):
            n = len(s)
            if n == 0:
                continue
            if n == 1:
                cv2.circle(frame, (int(ix[s[0]]), int(iy[s[0]])),
                           max(1, thick // 2), col, -1, cv2.LINE_AA)
                nseg += 1
                continue
            xi = ix[s]
            yi = iy[s]
            cxd = xi // cell
            cyd = yi // cell
            buckets = {}
            for m in range(n):
                buckets.setdefault((cxd[m], cyd[m]), []).append(m)
            used = np.zeros(n, bool)
            for m in range(n):
                cx, cy = cxd[m], cyd[m]
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        arr = buckets.get((cx + ox, cy + oy))
                        if not arr:
                            continue
                        for q in arr:
                            if q <= m:
                                continue
                            ddx = int(xi[q]) - int(xi[m])
                            ddy = int(yi[q]) - int(yi[m])
                            if ddx * ddx + ddy * ddy <= tol * tol:
                                cv2.line(frame, (int(xi[m]), int(yi[m])),
                                         (int(xi[q]), int(yi[q])), col,
                                         thick, cv2.LINE_AA)
                                used[m] = used[q] = True
                                nseg += 1
            for m in range(n):
                if not used[m]:
                    cv2.circle(frame, (int(xi[m]), int(yi[m])),
                               max(1, thick // 2), col, -1, cv2.LINE_AA)
                    nseg += 1
    return nseg, len(ux)
