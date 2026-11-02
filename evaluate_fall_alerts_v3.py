
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import train_fall_classifier as training
import train_fall_classifier_v3 as v3


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

PREDICTIONS_CSV = RESULT_DIR / "ml_replay_predictions.csv"
ALERTS_CSV = RESULT_DIR / "ml_potential_alerts.csv"
FALSE_ALERTS_CSV = RESULT_DIR / "ml_false_alerts.csv"

# Same alert settings as V1 and V2.
PROBABILITY_THRESHOLD = 0.50
EVIDENCE_WINDOW_SECONDS = 0.50
MIN_POSITIVE_PREDICTIONS = 2
ALERT_COOLDOWN_SECONDS = 8.0

# An alert counts as detecting a fall if it happens
# during the fall transition or up to 3 seconds after.
DETECTION_GRACE_SECONDS = 3.0


# --------------------------------------------------
# Replay one video with MediaPipe and Random Forest
# --------------------------------------------------

def replay_video(video_path, model, fold):
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(training.POSE_MODEL)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(
            f"Cannot open video: {video_path.name}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise ValueError(
            f"Invalid video FPS: {video_path.name}"
        )

    history = deque()
    positive_times = deque()
    landmark_filter = training.LandmarkMedianFilter()

    predictions = []
    alerts = []

    frame_number = 0
    last_sample_time = -1.0
    last_timestamp_ms = -1
    last_alert_time = float("-inf")

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as detector:

            while True:
                success, frame = cap.read()

                if not success:
                    break

                timestamp = frame_number / fps
                frame_number += 1

                # Sample using video time, not processing time.
                if (
                    timestamp - last_sample_time
                    < 1 / training.SAMPLE_FPS
                ):
                    continue

                last_sample_time = timestamp

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = max(
                    int(round(timestamp * 1000)),
                    last_timestamp_ms + 1,
                )

                last_timestamp_ms = timestamp_ms

                result = detector.detect_for_video(
                    mp_image,
                    timestamp_ms,
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
                    positive_times.clear()
                    continue

                history.append((timestamp, features))

                while (
                    history
                    and timestamp - history[0][0]
                    > training.WINDOW_SECONDS
                ):
                    history.popleft()

                window = training.temporal_features(
                    history
                )

                if window is None:
                    continue

                # Predict fall probability.
                fall_probability = float(
                    model.predict_proba(
                        window.reshape(1, -1)
                    )[0, 1]
                )

                predicted_fall = int(
                    fall_probability >= PROBABILITY_THRESHOLD
                )

                predictions.append({
                    "video": video_path.name,
                    "timestamp_sec": round(timestamp, 2),
                    "fall_probability": fall_probability,
                    "predicted_fall": predicted_fall,
                    "fold": fold,
                })

                # ----------------------------------
                # Alert decision logic
                # ----------------------------------

                if predicted_fall:
                    positive_times.append(timestamp)

                while (
                    positive_times
                    and timestamp - positive_times[0]
                    > EVIDENCE_WINDOW_SECONDS
                ):
                    positive_times.popleft()

                enough_evidence = (
                    len(positive_times)
                    >= MIN_POSITIVE_PREDICTIONS
                )

                cooldown_finished = (
                    timestamp - last_alert_time
                    >= ALERT_COOLDOWN_SECONDS
                )

                if enough_evidence and cooldown_finished:
                    alerts.append({
                        "video": video_path.name,
                        "alert_sec": round(timestamp, 2),
                        "fall_probability": fall_probability,
                        "fold": fold,
                    })

                    last_alert_time = timestamp

                    # New evidence is required after an alert.
                    positive_times.clear()

    finally:
        cap.release()

    return predictions, alerts


# --------------------------------------------------
# Match alerts against ground-truth fall events
# --------------------------------------------------

def evaluate_events(labels, alerts):
    fall_events = labels[
        labels["label"] == "FALL_TRANSITION"
    ].sort_values(
        ["video", "start_sec"]
    )

    # Only match each alert to one fall event.
    used_alerts = set()

    detected = 0
    delays = []

    for _, event in fall_events.iterrows():
        candidates = []

        for index, alert in enumerate(alerts):
            if index in used_alerts:
                continue

            if alert["video"] != event["video"]:
                continue

            alert_time = alert["alert_sec"]

            if (
                event["start_sec"]
                <= alert_time
                <= event["end_sec"]
                + DETECTION_GRACE_SECONDS
            ):
                candidates.append(index)

        if candidates:
            selected = min(
                candidates,
                key=lambda index: alerts[index]["alert_sec"],
            )

            used_alerts.add(selected)
            detected += 1

            delay = (
                alerts[selected]["alert_sec"]
                - event["start_sec"]
            )

            delays.append(delay)

    false_alerts = [
        alert
        for index, alert in enumerate(alerts)
        if index not in used_alerts
    ]

    actual_falls = len(fall_events)
    missed = actual_falls - detected

    event_recall = (
        detected / actual_falls
        if actual_falls > 0
        else 0.0
    )

    average_delay = (
        float(np.mean(delays))
        if delays
        else None
    )

    print("\nEVENT-LEVEL RESULTS")
    print("-" * 50)
    print("Actual fall events:", actual_falls)
    print("Detected fall events:", detected)
    print("Missed fall events:", missed)
    print("False alert episodes:", len(false_alerts))
    print(f"Fall event recall: {event_recall:.2%}")

    if average_delay is not None:
        print(
            f"Average detection delay: "
            f"{average_delay:.2f} seconds"
        )
    else:
        print("Average detection delay: N/A")

    return false_alerts


# --------------------------------------------------
# Five-fold out-of-fold event evaluation
# --------------------------------------------------

def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not training.POSE_MODEL.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe model: {training.POSE_MODEL}"
        )

    labels, videos = training.load_labels()

    video_names = sorted(
        labels["video"].unique()
    )

    video_targets = []

    for name in video_names:
        contains_fall = (
            (labels["video"] == name)
            & (labels["label"] == "FALL_TRANSITION")
        ).any()

        video_targets.append(
            int(contains_fall)
        )

    splitter = StratifiedKFold(
        n_splits=v3.N_FOLDS,
        shuffle=True,
        random_state=v3.RANDOM_STATE,
    )

    all_predictions = []
    all_alerts = []

    print("\nV3 FIVE-FOLD EVENT EVALUATION")
    print("=" * 50)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(
            video_names,
            video_targets,
        ),
        start=1,
    ):
        train_names = [
            video_names[i]
            for i in train_idx
        ]

        test_names = [
            video_names[i]
            for i in test_idx
        ]

        print(f"\nFOLD {fold}")
        print("-" * 50)
        print("Training videos:", len(train_names))
        print("Evaluation videos:", len(test_names))

        # Train this fold using only its 24
        # training videos.
        X_train = []
        y_train = []

        for name in train_names:
            features, targets = (
                training.extract_video(
                    videos[name],
                    labels,
                )
            )

            X_train.extend(features)
            y_train.extend(targets)

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"Fold {fold} contains only one class."
            )

        model = v3.create_model()
        model.fit(X_train, y_train)

        # Evaluate only this fold's unseen videos.
        for name in test_names:
            print(f"Processing: {name}")

            predictions, alerts = replay_video(
                videos[name],
                model,
                fold,
            )

            all_predictions.extend(predictions)
            all_alerts.extend(alerts)

            print(
                f"  Potential alerts: {len(alerts)}"
            )

    # ----------------------------------------
    # Event-level evaluation
    # ----------------------------------------

    false_alerts = evaluate_events(
        labels,
        all_alerts,
    )

    # ----------------------------------------
    # Save same 3 CSV outputs as V1 and V2
    # ----------------------------------------

    pd.DataFrame(
        all_predictions,
        columns=[
            "video",
            "timestamp_sec",
            "fall_probability",
            "predicted_fall",
            "fold",
        ],
    ).to_csv(
        PREDICTIONS_CSV,
        index=False,
        float_format="%.2f",
    )

    pd.DataFrame(
        all_alerts,
        columns=[
            "video",
            "alert_sec",
            "fall_probability",
            "fold",
        ],
    ).to_csv(
        ALERTS_CSV,
        index=False,
        float_format="%.2f",
    )

    pd.DataFrame(
        false_alerts,
        columns=[
            "video",
            "alert_sec",
            "fall_probability",
            "fold",
        ],
    ).to_csv(
        FALSE_ALERTS_CSV,
        index=False,
        float_format="%.2f",
    )

    print("\nResults saved in the results folder.")
    print("Saved:", PREDICTIONS_CSV.name)
    print("Saved:", ALERTS_CSV.name)
    print("Saved:", FALSE_ALERTS_CSV.name)


if __name__ == "__main__":
    main()
