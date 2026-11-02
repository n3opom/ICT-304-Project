
from pathlib import Path
from collections import deque
import math

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from statistics import median
from types import SimpleNamespace


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

VIDEO_DIR = BASE_DIR / "training_videos"
LABEL_FILE = BASE_DIR / "ICT304_Project_GroundTruthTable.xlsx"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_MODEL = MODEL_DIR / "fall_classifier.joblib"

SAMPLE_FPS = 30
WINDOW_SECONDS = 1.0
POST_FALL_POSITIVE_SECONDS = 0.6

# Existing MediaPipe pose model.
POSE_MODEL = MODEL_DIR / "pose_landmarker_full.task"

# Allowed labels.
FALL_LABELS = {"FALL_TRANSITION"}
POST_FALL_LABELS = {"POST_FALL"}

NON_FALL_LABELS = {
    "NORMAL",
    "STANDING",
    "WALKING",
    "SITTING",
    "CROUCHING",
    "CONTROLLED_LYING",
    "LYING_NO_FALL",
    "RECOVERY",
    "NON_FALL",
    "BENDING",
}

ALL_LABELS = (
    FALL_LABELS
    | POST_FALL_LABELS
    | NON_FALL_LABELS
)


# --------------------------------------------------
# Validate ground-truth workbook
# --------------------------------------------------

class LandmarkMedianFilter:
    """
    Three-frame median preprocessing.

    Matches the median-filter approach used
    by the live Random Forest application.
    """

    def __init__(self):
        self.history = deque(maxlen=3)

    def reset(self):
        self.history.clear()

    def update(self, landmarks):
        if not landmarks:
            self.reset()
            return None

        if (
            self.history
            and len(self.history[-1]) != len(landmarks)
        ):
            self.reset()

        coordinates = [
            (
                float(point.x),
                float(point.y),
                float(getattr(point, "z", 0.0)),
            )
            for point in landmarks
        ]

        self.history.append(coordinates)

        enough_samples = len(self.history) == 3

        filtered = []

        for i, point in enumerate(landmarks):
            if enough_samples:
                x = median(
                    frame[i][0] for frame in self.history
                )
                y = median(
                    frame[i][1] for frame in self.history
                )
                z = median(
                    frame[i][2] for frame in self.history
                )
            else:
                x, y, z = coordinates[i]

            filtered.append(
                SimpleNamespace(
                    x=x,
                    y=y,
                    z=z,
                    visibility=getattr(
                        point, "visibility", None
                    ),
                    presence=getattr(
                        point, "presence", None
                    ),
                )
            )

        return filtered

def load_labels():
    df = pd.read_excel(LABEL_FILE)

    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
    )

    required = {"video", "start_sec", "end_sec", "label"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {required - set(df.columns)}"
        )

    df = df.dropna(
        subset=["video", "start_sec", "end_sec", "label"]
    ).copy()

    df["video"] = df["video"].astype(str).str.strip()
    df["label"] = (
        df["label"].astype(str).str.strip().str.upper()
    )

    for column in ["start_sec", "end_sec"]:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        )

    invalid = df[
        (df["start_sec"] < 0)
        | (df["end_sec"] <= df["start_sec"])
    ]

    if not invalid.empty:
        raise ValueError(
            "Invalid timestamp intervals:\n"
            + invalid.to_string(index=False)
        )

    unknown = set(df["label"]) - ALL_LABELS

    if unknown:
        raise ValueError(
            f"Unknown labels: {sorted(unknown)}. "
            "Review the label mapping in this script."
        )

    videos = {
        file.name: file
        for file in VIDEO_DIR.glob("*.mp4")
    }

    missing = set(df["video"]) - set(videos)

    if missing:
        raise FileNotFoundError(
            f"Videos referenced but missing: {sorted(missing)}"
        )

    for video_name, rows in df.groupby("video"):
        rows = rows.sort_values("start_sec")
        previous_end = -1

        for _, row in rows.iterrows():
            if row["start_sec"] < previous_end - 0.001:
                raise ValueError(
                    f"Overlapping intervals in {video_name}"
                )
            previous_end = row["end_sec"]

        cap = cv2.VideoCapture(str(videos[video_name]))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video_name}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        duration = count / fps if fps > 0 else 0

        if previous_end > duration + 0.1:
            raise ValueError(
                f"{video_name}: last label ends at "
                f"{previous_end:.3f}s, video duration "
                f"is {duration:.3f}s"
            )

    print(f"Validated {len(videos)} video files.")
    print(f"Validated {len(df)} labelled intervals.")
    print("Labels:", df["label"].value_counts().to_dict())

    return df, videos


# --------------------------------------------------
# Extract body movement features
# --------------------------------------------------

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


# --------------------------------------------------
# Convert event labels to binary targets
# --------------------------------------------------

def get_target(video_rows, timestamp):
    matching = video_rows[
        (video_rows["start_sec"] <= timestamp)
        & (timestamp < video_rows["end_sec"])
    ]

    if matching.empty:
        return None

    label = matching.iloc[0]["label"]

    if label in FALL_LABELS:
        return 1

    if label in POST_FALL_LABELS:
        # Only classify the immediate post-fall period
        # as a positive fall-detection window.
        start = matching.iloc[0]["start_sec"]

        if timestamp - start <= POST_FALL_POSITIVE_SECONDS:
            return 1

        return 0

    return 0


# --------------------------------------------------
# Process video through MediaPipe
# --------------------------------------------------

def extract_video(video_path, labels):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(POSE_MODEL)
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    video_rows = labels[
        labels["video"] == video_path.name
    ]

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise ValueError(
            f"Cannot determine FPS: {video_path.name}"
        )

    X, y = [], []
    history = deque()
    landmark_filter = LandmarkMedianFilter()

    frame_number = 0
    last_sample_time = -1.0
    last_timestamp_ms = -1

    with PoseLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp = frame_number / fps
            frame_number += 1

            # Sample according to video time.
            if timestamp - last_sample_time < 1 / SAMPLE_FPS:
                continue

            last_sample_time = timestamp

            target = get_target(video_rows, timestamp)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = max(
                int(round(timestamp * 1000)),
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            landmarks = (
                result.pose_landmarks[0]
                if result.pose_landmarks
                else None
            )

            filtered_landmarks = landmark_filter.update(
                landmarks
            )

            features = frame_features(filtered_landmarks)
            
            if features is None:
                history.clear()
                continue

            history.append((timestamp, features))

            while (
                history
                and timestamp - history[0][0] > WINDOW_SECONDS
            ):
                history.popleft()

            window = temporal_features(history)

            if window is not None and target is not None:
                X.append(window)
                y.append(target)

    cap.release()

    print(
        f"{video_path.name}: "
        f"{len(X)} labelled movement windows"
    )

    return X, y


# --------------------------------------------------
# Train and evaluate
# --------------------------------------------------

def main():
    if not POSE_MODEL.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {POSE_MODEL}"
        )

    labels, videos = load_labels()

    X = []
    y = []
    groups = []

    for video_name in sorted(labels["video"].unique()):
        video_path = videos[video_name]

        features, targets = extract_video(
            video_path, labels
        )

        X.extend(features)
        y.extend(targets)
        groups.extend([video_name] * len(targets))

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if len(X) == 0 or len(np.unique(y)) < 2:
        raise ValueError(
            "Insufficient labelled fall and non-fall windows."
        )

    # Never split adjacent windows from the same video
    # between training and evaluation.
    splitter = GroupShuffleSplit(
        n_splits=30,
        test_size=0.25,
        random_state=42
    )

    selected_split = None

    for train_idx, test_idx in splitter.split(X, y, groups):
        if (
            len(np.unique(y[train_idx])) == 2
            and len(np.unique(y[test_idx])) == 2
        ):
            selected_split = (train_idx, test_idx)
            break

    if selected_split is None:
        raise ValueError(
            "Cannot create a video-group split "
            "with both classes in training and testing."
        )

    train_idx, test_idx = selected_split

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X[train_idx], y[train_idx])

    predictions = model.predict(X[test_idx])

    print("\nEVALUATION")
    print(
        classification_report(
            y[test_idx],
            predictions,
            labels=[0, 1],
            target_names=["NON_FALL", "FALL"],
            zero_division=0,
        )
    )

    print("CONFUSION MATRIX")
    print(confusion_matrix(
        y[test_idx], predictions, labels=[0, 1]
    ))

    print("\nTEST VIDEOS")
    for name in sorted(set(groups[test_idx])):
        print(name)

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump({
        "model": model,
        "sample_fps": SAMPLE_FPS,
        "window_seconds": WINDOW_SECONDS,
        "feature_version": 1,
    }, OUTPUT_MODEL)

    print(f"\nSaved model: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
