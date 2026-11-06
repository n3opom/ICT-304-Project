
import math
import numpy as np


KEYPOINTS = [0, 11, 12, 23, 24, 25, 26, 27, 28]


def frame_features(landmarks):
    if landmarks is None:
        return None

    points = np.array([
        [landmarks[i].x, landmarks[i].y]
        for i in KEYPOINTS
    ], dtype=float)

    visibility = np.array([
        getattr(landmarks[i], "visibility", 1.0)
        for i in KEYPOINTS
    ], dtype=float)

    shoulder = points[[1, 2]].mean(axis=0)
    hip = points[[3, 4]].mean(axis=0)
    knee = points[[5, 6]].mean(axis=0)

    torso = hip - shoulder

    angle = math.degrees(
        math.atan2(abs(torso[1]), abs(torso[0]))
    )

    valid = visibility >= 0.5

    if valid.any():
        visible_points = points[valid]
        span = visible_points.max(axis=0) - visible_points.min(axis=0)
    else:
        span = np.zeros(2)

    return np.array([
        angle / 90.0,
        shoulder[0],
        shoulder[1],
        hip[0],
        hip[1],
        knee[0],
        knee[1],
        float(span[0]),
        float(span[1]),
        float(visibility.mean()),
        float(valid.mean()),
    ], dtype=float)


def temporal_features(history):
    if len(history) < 5:
        return None

    times = np.array([item[0] for item in history])
    values = np.stack([item[1] for item in history])

    duration = times[-1] - times[0]

    if duration <= 0:
        return None

    dt = np.diff(times)
    differences = np.diff(values, axis=0)

    valid_dt = dt > 0

    if not valid_dt.any():
        return None

    velocity = (
        differences[valid_dt]
        / dt[valid_dt, None]
    )

    # Features describing posture and motion.
    return np.concatenate([
        values[-1],
        values.mean(axis=0),
        values.std(axis=0),
        values[-1] - values[0],
        np.abs(velocity).mean(axis=0),
        np.abs(velocity).max(axis=0),
    ])
