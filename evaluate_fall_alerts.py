
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

import train_fall_classifier as training


# ----------------------------------------
# Experiment settings
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "training_videos"
LABEL_FILE = BASE_DIR / "ICT304_Project_GroundTruthTable.xlsx"
MODEL_FILE = BASE_DIR / "models" / "fall_classifier.joblib"
POSE_MODEL = BASE_DIR / "models" / "pose_landmarker_full.task"
RESULT_DIR = BASE_DIR / "results"

# Exactly the videos held out during the
# training run you shared.
TEST_VIDEOS = [
    "Laying Down Pos 8.mp4",
    "Laying Down Pos 9.mp4",
    "Simulated Fall 13.mp4",
    "Simulated Fall 3.mp4",
    "Simulated Fall 9.mp4",
    "Sitting Multi 1.mp4",
    "Slow Laying Down 3.mp4",
    "Standing 1.mp4",
]

PROBABILITY_THRESHOLD = 0.50
REQUIRED_POSITIVES = 2
EVIDENCE_SECONDS = 0.50
ALERT_COOLDOWN_SECONDS = 8.0
DETECTION_GRACE_SECONDS = 3.0


def make_detector():
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(POSE_MODEL)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    )


def replay_video(path, model, sample_fps, window_seconds):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS: {path}")

    history = deque()
    evidence = deque()
    landmark_filter = training.LandmarkMedianFilter()
    rows = []
    alerts = []

    frame_number = 0
    last_sample_time = -1.0
    last_timestamp_ms = -1
    last_alert_time = float("-inf")

    fall_class_index = list(model.classes_).index(1)

    with make_detector() as detector:
        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp = frame_number / fps
            frame_number += 1

            if timestamp - last_sample_time < 1 / sample_fps:
                continue

            last_sample_time = timestamp

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = max(
                int(round(timestamp * 1000)),
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(
                image, timestamp_ms
            )

            landmarks = (
                result.pose_landmarks[0]
                if result.pose_landmarks
                else None
            )

            filtered_landmarks = landmark_filter.update(
                landmarks
            )

            features = training.frame_features(
                filtered_landmarks
            )

            if features is None:
                history.clear()
                evidence.clear()

                rows.append({
                    "video": path.name,
                    "time_sec": round(timestamp, 2),
                    "fall_probability": np.nan,
                    "potential_alert": False,
                })
                continue

            history.append((timestamp, features))

            while (
                history
                and timestamp - history[0][0] > window_seconds
            ):
                history.popleft()

            window = training.temporal_features(history)

            if window is None:
                continue

            probability = float(
                model.predict_proba(
                    window.reshape(1, -1)
                )[0, fall_class_index]
            )

            # Only retain recent positive predictions.
            while (
                evidence
                and timestamp - evidence[0] > EVIDENCE_SECONDS
            ):
                evidence.popleft()

            if probability >= PROBABILITY_THRESHOLD:
                evidence.append(timestamp)

            trigger = (
                len(evidence) >= REQUIRED_POSITIVES
                and timestamp - last_alert_time
                >= ALERT_COOLDOWN_SECONDS
            )

            if trigger:
                alerts.append({
                    "video": path.name,
                    "alert_sec": round(timestamp, 2),
                    "fall_probability": probability,
                })

                last_alert_time = timestamp
                evidence.clear()

            rows.append({
                "video": path.name,
                "time_sec": round(timestamp, 2),
                "fall_probability": probability,
                "potential_alert": trigger,
            })

    cap.release()
    return rows, alerts


def evaluate_events(labels, alerts):
    events = labels[
        labels["label"] == "FALL_TRANSITION"
    ].copy()

    matched_alerts = set()
    detected = 0
    delays = []

    for _, event in events.iterrows():
        matching = [
            i for i, alert in enumerate(alerts)
            if i not in matched_alerts
            and alert["video"] == event["video"]
            and event["start_sec"]
            <= alert["alert_sec"]
            <= event["end_sec"] + DETECTION_GRACE_SECONDS
        ]

        if matching:
            i = matching[0]
            matched_alerts.add(i)
            detected += 1

            delays.append(
                alerts[i]["alert_sec"]
                - event["start_sec"]
            )

    false_alerts = [
        alert for i, alert in enumerate(alerts)
        if i not in matched_alerts
    ]

    total = len(events)

    print("\nEVENT-LEVEL RESULTS")
    print("-" * 50)
    print(f"Actual fall events: {total}")
    print(f"Detected fall events: {detected}")
    print(f"Missed fall events: {total - detected}")
    print(f"False alert episodes: {len(false_alerts)}")

    if total:
        print(f"Fall event recall: {detected / total:.2%}")

    if delays:
        print(
            "Average detection delay: "
            f"{np.mean(delays):.2f} seconds"
        )

    return false_alerts


def main():
    RESULT_DIR.mkdir(exist_ok=True)

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]

    sample_fps = bundle["sample_fps"]
    window_seconds = bundle["window_seconds"]

    labels, videos = training.load_labels()

    labels = labels[
        labels["video"].isin(TEST_VIDEOS)
    ].copy()

    missing = set(TEST_VIDEOS) - set(videos)

    if missing:
        raise FileNotFoundError(
            f"Missing held-out recordings: {sorted(missing)}"
        )

    predictions = []
    all_alerts = []

    for name in TEST_VIDEOS:
        print(f"Processing: {name}")

        rows, alerts = replay_video(
            videos[name],
            model,
            sample_fps,
            window_seconds,
        )

        predictions.extend(rows)
        all_alerts.extend(alerts)

        print(f"  Potential alerts: {len(alerts)}")

    false_alerts = evaluate_events(
        labels, all_alerts
    )

    pd.DataFrame(predictions).to_csv(
        RESULT_DIR / "ml_replay_predictions.csv",
        index=False,
        float_format="%.2f",
    )

    pd.DataFrame(
        all_alerts,
        columns=[
            "video",
            "alert_sec",
            "fall_probability",
        ],
    ).to_csv(
        RESULT_DIR / "ml_potential_alerts.csv",
        index=False,
        float_format="%.2f",
    )

    pd.DataFrame(
        false_alerts,
        columns=[
            "video",
            "alert_sec",
            "fall_probability",
        ],
    ).to_csv(
        RESULT_DIR / "ml_false_alerts.csv",
        index=False,
        float_format="%.2f",
    )

    print("\nResults saved in the results folder.")


if __name__ == "__main__":
    main()
