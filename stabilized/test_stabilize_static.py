"""Integration checks against known subpixel camera motion and synthetic occlusions."""
from pathlib import Path

import cv2
import numpy as np

from stabilize_static import fit_transform, project, track, track_checkpoints


def main():
    data = np.load(Path(__file__).parent/'14_stabilized'/'reference.npz')
    ref, points, checks = data['background'], data['training'], data['validation']
    h, w = ref.shape
    grid = np.float32([[x, y] for x in np.linspace(.25*w, .75*w, 9)
                      for y in np.linspace(.20*h, .80*h, 5)])
    cases = [np.array([[1., 0., .37], [0., 1., -.63], [0., 0., 1.]]),
             np.array([[1.0002, -.003, 11.27], [.003, 1.0002, -4.38], [0., 0., 1.]]),
             np.array([[1.0004, -.002, -6.43], [.001, .9995, 5.24], [3e-7, -5e-7, 1.]])]
    for k, truth in enumerate(cases):
        current = cv2.warpPerspective(ref, truth, (w, h), flags=cv2.INTER_CUBIC)
        # Occlusions affect both camera estimation and independent checkpoint availability.
        cv2.rectangle(current, (450, 150), (810, 530), 210, -1)
        cv2.rectangle(current, (1650, 430), (1880, 700), 10, -1)
        q, good, _, _ = track(ref, current, points, np.eye(3))
        estimated, n, _ = fit_transform(points[good], q[good], ref.shape, 'homography')
        error = np.linalg.norm(project(grid, estimated)-project(grid, truth), axis=1)
        observed, valid = track_checkpoints(ref, current, checks, estimated, data['stable'])
        measurement_error = np.linalg.norm(observed[valid]-project(checks[valid], truth), axis=1)
        assert n > 100, n
        assert error.max() < .15, error
        assert valid.sum() >= 10, valid.sum()
        assert np.median(measurement_error) < .20, measurement_error
        recovered = project(project(grid, truth), np.linalg.inv(truth))
        assert np.max(np.abs(recovered-grid)) < .001
        print(f'case {k}: max camera error={error.max():.4f}px, '
              f'median independent measurement error={np.median(measurement_error):.4f}px, '
              f'visible checkpoints={valid.sum()}', flush=True)


if __name__ == '__main__':
    main()
