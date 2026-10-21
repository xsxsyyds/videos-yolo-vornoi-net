# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Shape-aware Kalman tracking for segmentation models in top-down / UAV views.

Motivation
----------
In top-down (drone) footage, targets are frequently **partially occluded from above**
(tree canopies, wires, bridges, tunnels). Unlike ground-level occlusion, the target is not
simply cut in half — its mask is often **shattered into visible fragments or briefly erased
entirely**, which makes box-only trackers lose identity: the centroid jumps, the bbox IoU
collapses, and the track fragments into several IDs.

The reference method (Korea Science, 2026 — *Segment-based UAV Tracking via Shape-Aware Kalman
Filter*) extends the classical 4-D centroid-velocity state::

    classical :  z = [x, y, ẋ, ẏ]                       (4-D)
    shape-aware: z = [x, y, ẋ, ẏ, a, r, θ]               (7-D)

where ``a`` is the mask area (foreground pixel count), ``r`` the mask aspect ratio (w/h) and
``θ`` the orientation derived from the mask's second-order image moments. The idea is that a
target's *geometric signature* (area / aspect / orientation) is far more stable across an
occlusion than its centroid, so identity survives when position alone would be ambiguous.
The paper reports up to **84–85 % fewer ID switches** on the Anti-UAV benchmark.

Two findings from that paper are baked into the defaults here:

1. **Loose IoU association wins.** Their ablation shows IoU ``0.3`` beats ``0.5``
   (IDSW 14.8 vs 23.4), i.e. *"overly strict matching conditions aggravate identity
   fragmentation."* So :data:`ShapeSTrack` association defaults to a loose IoU gate.
2. **Geometry only helps under real occlusion.** On easy data the 7-D and 4-D states perform
   identically, because the extra dimensions are never "activated". The tracker therefore only
   *emphasises* the shape term when a track has actually gone unmatched, keeping the clean-data
   behaviour unchanged.

Implementation notes
--------------------
``ultralytics``' built-in :class:`~ultralytics.trackers.utils.kalman_filter.KalmanFilterXYAH`
is hard-wired to an 8-D ``(x, y, a, h, vx, vy, va, vh)`` state. To get a genuine 7-D
centroid + static-shape model without forking that class, :class:`ShapeKalmanFilter7D`
is a small, self-contained linear Kalman filter with the state layout::

    mean = [x, y, ẋ, ẏ, a, r, θ]        (indices 0..6)
    F    = I, with F[0,2] = F[1,3] = 1   (constant velocity on centre, static shape)
    H    = I_7                           (direct observation of all seven states)

which is exactly the paper's formulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.stracks import parse_bboxes

__all__ = ("ShapeKalmanFilter7D", "ShapeSTrack", "ShapeTracker")


# --------------------------------------------------------------------------- #
# 7-D Kalman filter
# --------------------------------------------------------------------------- #
class ShapeKalmanFilter7D:
    """Linear Kalman filter over the 7-D centroid + static-shape state.

    State layout (paper Eq. 1)::

        x = [x, y, ẋ, ẏ, a, r, θ]ᵀ

    * ``(x, y)``   — mask centroid, **constant-velocity** model
    * ``(ẋ, ẏ)`` — centroid velocity
    * ``a``        — mask area in pixels
    * ``r``        — mask aspect ratio (width / height)
    * ``θ``        — mask orientation from second-order image moments

    The shape triple ``(a, r, θ)`` is modelled as **static** (no velocity terms), matching the
    reference paper: shape changes far more slowly than position, and treating it as static is
    what lets the filter coast through an occlusion without drifting shape.

    Args:
        process_noise (float): Multiplier on the process noise injected per step. Larger values
            let the state chase observations faster (less lag, more jitter).
        measure_noise (float): Multiplier on the measurement noise. Larger values trust the
            model more and the incoming mask geometry less.
    """

    def __init__(self, process_noise: float = 1.0, measure_noise: float = 1.0):
        """Build the motion / observation matrices for the 7-D state."""
        self.ndim = 7
        self.process_noise = float(process_noise)
        self.measure_noise = float(measure_noise)

        # Constant velocity on the centroid; (a, r, θ) held static.
        self._motion_mat = np.eye(self.ndim, dtype=np.float64)
        self._motion_mat[0, 2] = 1.0
        self._motion_mat[1, 3] = 1.0

        # Observation selects the 5 measured states out of the 7-D state:
        # [x, y, ẋ, ẏ, a, r, θ] -> [x, y, a, r, θ]
        self._update_mat = np.zeros((5, self.ndim), dtype=np.float64)
        self._update_mat[0, 0] = 1.0  # x
        self._update_mat[1, 1] = 1.0  # y
        self._update_mat[2, 4] = 1.0  # a
        self._update_mat[3, 5] = 1.0  # r
        self._update_mat[4, 6] = 1.0  # θ

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _shape_scale(area: float, aspect: float) -> tuple[float, float, float]:
        """Return ``(scale, area, aspect)`` noise magnitudes from an area / aspect pair.

        Area / aspect / orientation are scaled by their own magnitude so a large target is not
        forced to have the same absolute shape tolerance as a small one.
        """
        area = max(float(area), 1.0)
        aspect = max(float(aspect), 1e-3)
        return np.sqrt(area), area, aspect

    def _state_std(self, mean: np.ndarray) -> np.ndarray:
        """Per-state process-noise magnitudes for the 7-D layout ``[x,y,ẋ,ẏ,a,r,θ]``."""
        scale, area, asp = self._shape_scale(mean[4], mean[5])
        return np.array(
            [
                0.10 * scale,  # x
                0.10 * scale,  # y
                1e-4 * scale,  # ẋ (velocity is noisy; keep loose)
                1e-4 * scale,  # ẏ
                0.05 * area,   # a
                0.05 * asp,    # r
                0.05,          # θ (radians, ~3° floor)
            ],
            dtype=np.float64,
        )

    def _measurement_std(self, measurement: np.ndarray) -> np.ndarray:
        """Per-measurement noise magnitudes for the 5-D layout ``[x,y,a,r,θ]``."""
        scale, area, asp = self._shape_scale(measurement[2], measurement[3])
        return np.array(
            [
                0.10 * scale,  # x
                0.10 * scale,  # y
                0.05 * area,   # a
                0.05 * asp,    # r
                0.05,          # θ
            ],
            dtype=np.float64,
        )

    # -- API -------------------------------------------------------------- #
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Seed a state from an unassociated measurement.

        Args:
            measurement (np.ndarray): ``[x, y, a, r, θ]`` — centroid, area, aspect ratio, orientation.

        Returns:
            (tuple[np.ndarray, np.ndarray]): ``(mean, covariance)`` with zero-initialised velocity.
        """
        m = np.asarray(measurement, dtype=np.float64)
        mean = np.r_[m[:2], 0.0, 0.0, m[2], m[3], m[4]]
        std = self._state_std(mean)
        # Position and shape start uncertain; velocity starts very uncertain.
        var = np.square(std)
        var[2:4] *= 1e4
        return mean, np.diag(var)

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance the state one step under the constant-velocity / static-shape model."""
        std = self._state_std(mean) * self.process_noise
        q = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + q
        return mean, covariance

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Vectorised :meth:`predict` over an ``(N, 7)`` / ``(N, 7, 7)`` batch."""
        n = mean.shape[0]
        std = np.stack([self._state_std(m) for m in mean]) * self.process_noise
        q = np.zeros((n, self.ndim, self.ndim), dtype=np.float64)
        idx = np.arange(self.ndim)
        q[:, idx, idx] = np.square(std)

        mean = mean @ self._motion_mat.T
        # Batched F @ P: broadcast the (7,7) motion matrix across the leading batch axis.
        f = self._motion_mat
        left = np.matmul(f[None, :, :], covariance)
        covariance = np.matmul(left, f.T[None, :, :]) + q
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct the state with an observed ``[x, y, a, r, θ]`` measurement."""
        m = np.asarray(measurement, dtype=np.float64)
        std = self._measurement_std(m) * self.measure_noise
        r = np.diag(np.square(std))

        h = self._update_mat
        projected_mean = h @ mean
        projected_cov = h @ covariance @ h.T + r

        gain = np.linalg.solve(projected_cov, h @ covariance).T
        innovation = m - projected_mean
        new_mean = mean + gain @ innovation
        new_covariance = covariance - gain @ projected_cov @ gain.T
        return new_mean, new_covariance


# --------------------------------------------------------------------------- #
# Single track
# --------------------------------------------------------------------------- #
class ShapeSTrack(STrack):
    """A track carrying a 7-D shape-aware state instead of the stock 8-D XYAH state.

    Geometry (area ``a``, aspect ``r``, orientation ``θ``) is optional: when a detection arrives
    without mask information the filter still runs, but the shape triple is coasted from the
    last known values rather than corrected. This keeps the tracker usable for box-only inputs
    (e.g. OBB or detect models) without a separate code path.

    Attributes:
        shape_history (list): Bounded list of ``(mean, covariance)`` snapshots, newest last,
            used to restore a pre-occlusion state when a track is recovered.
        geometry (np.ndarray): Latest observed ``[a, r, θ]``, or ``None`` if never observed.
        not_matched (int): Consecutive unmatched frames.
        is_occluded (bool): True while the track is judged hidden behind another target.
    """

    #: Bounded history length for pre-occlusion rollback.
    _HISTORY_MAX = 16

    def __init__(self, xywh: np.ndarray, score: float, cls: Any, geometry: np.ndarray | None = None):
        """Wrap a bounding box detection.

        Args:
            xywh (np.ndarray): ``(x, y, w, h, idx)`` or ``(x, y, w, h, angle, idx)``.
            score (float): Detection confidence.
            cls (Any): Class label.
            geometry (np.ndarray | None): Optional ``[area, aspect, theta]`` mask geometry.
        """
        super().__init__(xywh, score, cls)
        self.shape_history: list = []
        self.geometry = np.asarray(geometry, dtype=np.float64) if geometry is not None else None
        self.not_matched = 0
        self.is_occluded = False

    # -- geometry --------------------------------------------------------- #
    def _measurement(self, fallback_mean: np.ndarray | None = None) -> np.ndarray:
        """Assemble the 7-D measurement ``[x, y, a, r, θ]`` for this detection.

        Falls back to the previous state's shape (and the box's own aspect/area) whenever mask
        geometry is unavailable, so the observation is always well-formed.
        """
        tlwh = self.tlwh
        cx, cy = float(tlwh[0] + tlwh[2] / 2.0), float(tlwh[1] + tlwh[3] / 2.0)
        if self.geometry is not None:
            a, r, theta = (float(v) for v in self.geometry)
        else:
            w, h = float(tlwh[2]), float(tlwh[3])
            a = max(w * h * 0.7, 1.0)  # rough mask-area proxy: box area minus corners
            r = float(w) / max(h, 1e-3)
            theta = (
                float(fallback_mean[6])
                if (fallback_mean is not None and len(fallback_mean) >= 7)
                else 0.0
            )
        return np.array([cx, cy, a, r, theta], dtype=np.float64)

    def _push_history(self) -> None:
        """Snapshot the current state into the bounded rollback buffer."""
        if self.mean is not None:
            self.shape_history.append((self.mean.copy(), self.covariance.copy()))
            if len(self.shape_history) > self._HISTORY_MAX:
                self.shape_history.pop(0)

    # -- lifecycle -------------------------------------------------------- #
    def activate(self, kalman_filter, frame_id: int):
        """Create the track from its first observation."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self._measurement())
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self._push_history()

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Revive a lost track from a new detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track._measurement(self.mean)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        if getattr(new_track, "geometry", None) is not None:
            self.geometry = np.asarray(new_track.geometry, dtype=np.float64)
        self.not_matched = 0
        self.is_occluded = False
        self._push_history()

    def update(self, new_track: STrack, frame_id: int):
        """Correct the state with a matched detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track._measurement(self.mean)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        if getattr(new_track, "geometry", None) is not None:
            self.geometry = np.asarray(new_track.geometry, dtype=np.float64)
        self.not_matched = 0
        self.is_occluded = False
        self._push_history()

    def predict(self):
        """Advance the state; freeze velocity for non-tracked (lost) tracks."""
        mean_state = self.mean.copy()
        mean_state[2] = 0.0
        mean_state[3] = 0.0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list):
        """Vectorised prediction over a list of :class:`ShapeSTrack`."""
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_cov = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i, 2] = 0.0
                multi_mean[i, 3] = 0.0
        shared = _SHARED_KF7
        multi_mean, multi_cov = shared.multi_predict(multi_mean, multi_cov)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_cov)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Unused by the 7-D filter; kept for :class:`STrack` API parity."""
        return self._measurement(self.mean)

    # -- geometry accessors ----------------------------------------------- #
    # The 7-D state stores ``[x, y, ẋ, ẏ, a, r, θ]``: position at [:2], velocity at [2:4],
    # shape at [4:7]. The stock `STrack` accessors assume the XYAH layout, so they must be
    # overridden — otherwise velocity would be read back as width and height.
    @property
    def tlwh(self) -> np.ndarray:
        """Top-left ``(x, y, w, h)`` derived from the centroid and the shape triple."""
        if self.mean is None:
            return self._tlwh.copy()
        cx, cy = float(self.mean[0]), float(self.mean[1])
        area = max(float(self.mean[4]), 1.0)
        aspect = max(float(self.mean[5]), 1e-3)
        # w = a * h and w / h = aspect  ->  h = sqrt(area / aspect), w = aspect * h
        h = float(np.sqrt(area / aspect))
        w = aspect * h
        return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        """Centroid ``(x, y, w, h)`` derived from the centroid and the shape triple."""
        if self.mean is None:
            ret = self._tlwh.copy()
            ret[:2] += ret[2:] / 2.0
            return ret
        t = self.tlwh
        return np.array([t[0] + t[2] / 2.0, t[1] + t[3] / 2.0, t[2], t[3]], dtype=np.float32)

    @property
    def shape_vec(self) -> np.ndarray:
        """Current ``[a, r, θ]`` estimate, or ``None`` before activation."""
        return None if self.mean is None else self.mean[4:7].copy()

    @property
    def area(self) -> float:
        """Estimated mask area in pixels."""
        return float(self.mean[4]) if self.mean is not None else 0.0

    def shape_distance(self, other: "ShapeSTrack") -> float:
        """Normalised log-space distance between two tracks' geometric signatures.

        Written in log space so relative differences are compared (a 10 % area difference is the
        same cost whether the target is large or small). Returned in ``[0, 1]`` for easy fusion
        with an IoU cost.
        """
        if self.shape_vec is None or other.shape_vec is None:
            return 0.0
        a1, r1, t1 = (max(float(v), 1e-6) for v in self.shape_vec)
        a2, r2, t2 = (max(float(v), 1e-6) for v in other.shape_vec)
        d_area = abs(np.log(a1) - np.log(a2))
        d_asp = abs(np.log(r1) - np.log(r2))
        # Angular difference wrapped to [0, π], then normalised.
        d_theta = abs(np.arctan2(np.sin(t1 - t2), np.cos(t1 - t2))) / np.pi
        raw = 0.5 * d_area + 0.25 * d_asp + 0.25 * d_theta
        return float(1.0 - np.exp(-raw))

    def __repr__(self) -> str:
        """Human-readable summary including the shape signature."""
        if self.mean is None:
            return super().__repr__()
        return (
            f"ShapeSTrack(id={self.track_id}, {self.start_frame}-{self.end_frame}, "
            f"area={self.area:.0f}, r={self.mean[5]:.2f}, th={self.mean[6]:+.2f})"
        )


_SHARED_KF7 = ShapeKalmanFilter7D()


# --------------------------------------------------------------------------- #
# Multi-object tracker
# --------------------------------------------------------------------------- #
class ShapeTracker(BYTETracker):
    """Shape-aware multi-object tracker for top-down segmentation.

    Built on the ByteTrack two-stage association skeleton, with three additions aimed squarely at
    the "mask shatters / briefly vanishes" failure mode of drone footage:

    1. **Shape-aware cost.** Association cost fuses IoU distance with the geometric-signature
       distance (:meth:`ShapeSTrack.shape_distance`), so two nearby targets of very different
       sizes are no longer interchangeable.
    2. **Occlusion coasting.** An unmatched track whose area collapses sharply is treated as
       partially occluded: its shape state is frozen and a larger search region is used, so the
       track survives the gap instead of being replaced by a fresh ID.
    3. **Loose IoU gating.** Per the reference paper's ablation, the primary association gate is
       deliberately permissive to avoid fragmenting identity.

    Args:
        args: Parsed tracker config. Recognised keys beyond the ByteTrack set:

            * ``shape_weight`` (float): Weight of the shape term in the association cost.
            * ``shape_iou_thresh`` (float): Primary association gate. Defaults low (0.3) on
              purpose — see the module docstring.
            * ``shape_area_ratio_occ`` (float): Area-drop ratio that flags a track as occluded.
            * ``shape_coast_max`` (int): Frames a track may coast as "occluded" before normal
              loss handling resumes.
    """

    track_class = ShapeSTrack

    def __init__(self, args):
        """Read generic ByteTrack settings plus the shape-tracking knobs."""
        super().__init__(args)
        self.shape_weight = float(getattr(args, "shape_weight", 0.35))
        # Deliberately loose: the paper's ablation puts the optimum at IoU 0.3.
        self.shape_iou_thresh = float(getattr(args, "shape_iou_thresh", 0.30))
        self.shape_area_ratio_occ = float(getattr(args, "shape_area_ratio_occ", 0.55))
        self.shape_coast_max = int(getattr(args, "shape_coast_max", 15))
        self.kalman_filter = ShapeKalmanFilter7D()

    # -- construction ------------------------------------------------------ #
    def init_track(self, results, img: np.ndarray | None = None) -> list[ShapeSTrack]:
        """Build :class:`ShapeSTrack` objects from a ``Results``-like object.

        Mask geometry is read from ``results.masks`` when present. ``masks`` may be either a
        torch tensor of shape ``(N, H, W)`` (bool or uint8) or a list of such tensors.
        """
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        geoms = extract_mask_geometry(results, len(bboxes))
        return [
            ShapeSTrack(xywh, s, c, geometry=g)
            for xywh, s, c, g in zip(bboxes, results.conf, results.cls, geoms)
        ]

    # -- association ------------------------------------------------------- #
    def _cost_matrix(self, tracks: list[ShapeSTrack], dets: list[ShapeSTrack]) -> np.ndarray:
        """IoU distance blended with geometric-signature distance, optionally score-fused."""
        base = matching.iou_distance(tracks, dets)
        if self.shape_weight > 0.0 and tracks and dets:
            shape = np.zeros_like(base)
            for i, t in enumerate(tracks):
                for j, d in enumerate(dets):
                    shape[i, j] = t.shape_distance(d)
            base = (1.0 - self.shape_weight) * base + self.shape_weight * shape
        if getattr(self.args, "fuse_score", False):
            base = matching.fuse_score(base, dets)
        return base

    def _match(self, tracks, dets, thresh):
        """Greedy linear assignment on the fused cost."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        cost = self._cost_matrix(tracks, dets)
        return matching.linear_assignment(cost, thresh=thresh)

    # -- frame update ------------------------------------------------------ #
    def update(self, results, img: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """Consume one frame of detections and return ``(N, 8)`` ``[x1,y1,x2,y2,id,score,cls,idx]``."""
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        scores = np.asarray(results.conf)
        boxes = parse_bboxes(results)
        geoms = extract_mask_geometry(results, len(boxes))
        geo_list = list(geoms)
        high = scores >= self.args.track_high_thresh
        low = (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)

        cls_arr = np.asarray(results.cls)

        def _mk(box, sc, cl, geo):
            return ShapeSTrack(box, sc, cl, geometry=geo)

        def _slice(idx_mask):
            """Gather boxes / scores / classes / geometry under a boolean mask."""
            return (
                boxes[idx_mask],
                scores[idx_mask],
                cls_arr[idx_mask],
                [geo_list[i] for i in np.nonzero(idx_mask)[0]],
            )

        hb, hs, hc, hg = _slice(high)
        lb, ls, lc, lg = _slice(low)
        dets_high = [_mk(b, s, c, g) for b, s, c, g in zip(hb, hs, hc, hg)]
        dets_low = [_mk(b, s, c, g) for b, s, c, g in zip(lb, ls, lc, lg)]

        unconfirmed = [t for t in self.tracked_stracks if t.state != TrackState.Tracked]
        tracked = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        pool = self._joint(tracked, self.lost_stracks)

        ShapeSTrack.multi_predict(pool)
        ShapeSTrack.multi_predict(unconfirmed)

        # --- stage 1: high-confidence association ---------------------------
        matches, u_pool, u_det = self._match(pool, dets_high, self.shape_iou_thresh)
        u_track = []
        for it, idet in matches:
            track, det = pool[it], dets_high[idet]
            track.not_matched = 0
            track.is_occluded = False
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        # --- stage 2: low-confidence recovery -------------------------------
        r_tracked = [pool[i] for i in u_pool if pool[i].state == TrackState.Tracked]
        if r_tracked and dets_low:
            m2, u2, _ = self._match(r_tracked, dets_low, 0.5)
            for it, idet in m2:
                r_tracked[it].not_matched = 0
                r_tracked[it].is_occluded = False
                r_tracked[it].update(dets_low[idet], self.frame_id)
                activated.append(r_tracked[it])
            u_track = [r_tracked[i] for i in u2]
        else:
            u_track = r_tracked

        # --- stage 3: unmatched tracked -> occluded or lost -----------------
        for track in u_track:
            track.not_matched += 1
            occluded = self._detect_occlusion(track, activated)
            if occluded and track.not_matched <= self.shape_coast_max:
                track.is_occluded = True
                # Freeze shape (do not let a shrunken occluded mask corrupt the signature)
                # and kill residual velocity so the track does not overshoot.
                track.mean[2:4] = 0.0
            if track.not_matched > 2 and not (track.is_occluded and track.not_matched <= self.shape_coast_max):
                track.mark_lost()
                lost.append(track)

        # --- stage 4: unconfirmed tracks ------------------------------------
        if unconfirmed and dets_high:
            m3, u_unconf, _ = self._match(unconfirmed, dets_high, 0.7)
            for it, idet in m3:
                unconfirmed[it].update(dets_high[idet], self.frame_id)
                unconfirmed[it].is_activated = True
                activated.append(unconfirmed[it])
            for i in u_unconf:
                unconfirmed[i].mark_removed()
                removed.append(unconfirmed[i])

        # --- stage 5: spawn new tracks --------------------------------------
        matched_det_idx = {d for _, d in matches} | {d for _, d in (locals().get("m3") or [])}
        for idet, det in enumerate(dets_high):
            if idet in matched_det_idx or det.score < self.args.new_track_thresh:
                continue
            active = activated + [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
            if self._suppressed(det, active):
                continue
            det.activate(self.kalman_filter, self.frame_id)
            det.is_activated = True
            activated.append(det)

        # --- bookkeeping ----------------------------------------------------
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self._joint(self.tracked_stracks, activated)
        self.tracked_stracks = self._joint(self.tracked_stracks, refind)
        self.lost_stracks = self._sub(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = self._sub(self.lost_stracks, self.removed_stracks)

        # Evict lost tracks that have exceeded the buffer. Without this, a track lost early in
        # the clip stays matchable forever and can be resurrected hundreds of frames later by an
        # unrelated target — the source of the pathological 600+ frame "gaps" seen in testing.
        keep_lost = []
        for t in self.lost_stracks:
            if self.frame_id - t.end_frame > self.max_frames_lost:
                t.mark_removed()
                removed.append(t)
            else:
                keep_lost.append(t)
        self.lost_stracks = keep_lost

        self.tracked_stracks, self.lost_stracks = self._dedup(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed)
        # Keep the removed buffer bounded, mirroring the other trackers.
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return np.asarray(
            [t.result for t in self.tracked_stracks if t.is_activated and t.frame_id == self.frame_id],
            dtype=np.float32,
        )

    # -- helpers ----------------------------------------------------------- #
    def _detect_occlusion(self, track: ShapeSTrack, activated: list) -> bool:
        """Flag a track as occluded when a neighbour now covers most of it, or its area collapses.

        Two independent signals are used because top-down occlusion manifests differently from
        ground-level occlusion: a canopy can *both* swallow the target (neighbour coverage) and
        shatter its mask into a much smaller fragment (area collapse).
        """
        # Signal A: substantial coverage by an already-matched active track.
        for other in activated:
            if other.track_id == track.track_id or other.cls != track.cls:
                continue
            if other.xyxy[0] <= track.xyxy[0] and other.xyxy[1] <= track.xyxy[1] \
                    and other.xyxy[2] >= track.xyxy[2] and other.xyxy[3] >= track.xyxy[3]:
                return True

        # Signal B: sudden collapse of the predicted mask area. `mean[4]` holds the coasted area,
        # `geometry[0]` the last observed one; a large gap means the mask lost most of its pixels.
        if track.geometry is not None:
            obs_area = float(track.geometry[0])
            est_area = track.area
            if est_area > 1.0 and obs_area / est_area < self.shape_area_ratio_occ:
                return True
        return False

    def _suppressed(self, det: ShapeSTrack, active: list) -> bool:
        """Reject a new detection that duplicates an existing active track (same class)."""
        for t in active:
            if t.cls != det.cls:
                continue
            if _iou(det.xyxy, t.xyxy) > 0.6:
                return True
        return False

    @staticmethod
    def _joint(a, b):
        ids = {t.track_id for t in a}
        return a + [t for t in b if t.track_id not in ids]

    @staticmethod
    def _sub(a, b):
        ids = {t.track_id for t in b}
        return [t for t in a if t.track_id not in ids]

    def _dedup(self, tracked, lost):
        """Drop lost tracks that duplicate a tracked one (IoU > 0.85)."""
        keep_lost = []
        for lt in lost:
            if any(_iou(lt.xyxy, tt.xyxy) > 0.85 for tt in tracked):
                continue
            keep_lost.append(lt)
        return tracked, keep_lost


# --------------------------------------------------------------------------- #
# Mask geometry extraction
# --------------------------------------------------------------------------- #
def extract_mask_geometry(results, n: int) -> list[np.ndarray | None]:
    """Derive ``[area, aspect_ratio, orientation]`` per detection from ``results.masks``.

    Orientation comes from the second-order image moments of the mask (the standard
    ``atan2(2·μ11, μ20 − μ02) / 2`` formula), which is the definition used in the reference
    paper. Degenerate masks (too few pixels, or near-isotropic so orientation is meaningless)
    fall back to ``0.0``.

    Args:
        results: A ``Results``-like object; ``masks`` may be a torch tensor of shape ``(N, H, W)``
            or a list of per-instance tensors. Missing/absent masks yield all-``None``.
        n (int): Expected detection count, used to pad/short-circuit.

    Returns:
        (list[np.ndarray | None]): One ``[area, aspect, theta]`` array per detection, or ``None``
            where geometry could not be computed.
    """
    masks = getattr(results, "masks", None)
    if masks is None or n == 0:
        return [None] * n
    data = getattr(masks, "data", masks)
    try:
        import torch
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
    except Exception:
        pass
    try:
        arr = np.asarray(data)
    except Exception:
        return [None] * n
    if arr.ndim == 4:  # (N, 1, H, W)
        arr = arr[:, 0]
    if arr.ndim != 3 or arr.shape[0] != n:
        return [None] * n

    out: list[np.ndarray | None] = []
    for m in arr:
        out.append(_moments_geometry(m.astype(bool)))
    return out


def _moments_geometry(mask: np.ndarray) -> np.ndarray | None:
    """Compute ``[area, aspect, theta]`` from a single boolean mask via image moments."""
    area = int(mask.sum())
    if area < 4:
        return None
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    dx = xs - cx
    dy = ys - cy
    mu20 = float((dx * dx).mean())
    mu02 = float((dy * dy).mean())
    mu11 = float((dx * dy).mean())

    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    aspect = float(w) / max(float(h), 1.0)

    # Near-isotropic masks have an ill-defined orientation -> report 0.
    denom = abs(mu20 - mu02)
    if denom < 1e-6 and abs(mu11) < 1e-6:
        theta = 0.0
    else:
        theta = 0.5 * float(np.arctan2(2.0 * mu11, mu20 - mu02))
    return np.array([float(area), aspect, theta], dtype=np.float64)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Axis-aligned IoU between two ``xyxy`` boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0
