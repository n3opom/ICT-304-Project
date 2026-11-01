
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

import train_fall_classifier as training
from src.temporal_fall_detector import (
    TemporalFallDetector,
    FallState,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

PREDICTIONS_CSV = RESULT_DIR / "ml_replay_predictions.csv"
ALERTS_CSV = RESULT_DIR / "ml_potential_alerts.csv"
FALSE_ALERTS_CSV = RESULT_DIR / "ml_false_alerts.csv"

SAMPLE_FPS = training.SAMPLE_FPS

ALERT_COOLDOWN_SECONDS = 8.0
DETECTION_GRACE_SECONDS = 3.0


# --------------------------------------------------
# Replay video using the rule-based detector
# --------------------------------------------------

def replay_video(video_path):
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
            f"Invalid FPS: {video_path.name}"
        )

    fall_detector = TemporalFallDetector()

    predictions = []
    alerts = []

    frame_number = 0
    last_sample_time = -1.0
    last_timestamp_ms = -1
    last_alert_time = float("-inf")

    previous_alert_state = False

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

                if (
                    timestamp - last_sample_time
                    < 1 / SAMPLE_FPS
                ):
                    continue

                last_sample_time = timestamp

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
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

                # Run the original temporal algorithm.
                fall_state, posture = fall_detector.update(
                    landmarks,
                    timestamp=timestamp,
                )

                # Evaluate a potential fall, not only
                # a fully confirmed FALL state.
                alert_state = fall_state in (
                    FallState.POSSIBLE_FALL,
                    FallState.FALL,
                )

                predicted_fall = int(alert_state)

                predictions.append({
                    "video": video_path.name,
                    "timestamp_sec": round(timestamp, 2),
                    "fall_probability": np.nan,
                    "predicted_fall": predicted_fall,
                    "fall_state": fall_state.value,
                    "posture": posture.value,
                })

                # ----------------------------------
                # Potential-fall alert logic
                # ----------------------------------

                # Only generate a candidate when
                # entering a potential-fall state.
                new_potential_fall = (
                    alert_state
                    and not previous_alert_state
                )

                cooldown_finished = (
                    timestamp - last_alert_time
                    >= ALERT_COOLDOWN_SECONDS
                )

                if (
                    new_potential_fall
                    and cooldown_finished
                ):
                    alerts.append({
                        "video": video_path.name,
                        "alert_sec": round(timestamp, 2),
                        "fall_probability": np.nan,
                        "fall_state": fall_state.value,
                    })

                    last_alert_time = timestamp

                previous_alert_state = alert_state

    finally:
        cap.release()

    return predictions, alerts


# --------------------------------------------------
# Compare alerts against annotated fall events
# --------------------------------------------------

def evaluate_events(labels, alerts):
    fall_events = labels[
        labels["label"] == "FALL_TRANSITION"
    ].sort_values(
        ["video", "start_sec"]
    )

    matched_alerts = set()
    detected_count = 0
    delays = []

    for _, event in fall_events.iterrows():
        candidates = []

        for index, alert in enumerate(alerts):
            if index in matched_alerts:
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
                key=lambda i: alerts[i]["alert_sec"],
            )

            matched_alerts.add(selected)
            detected_count += 1

            delay = (
                alerts[selected]["alert_sec"]
                - event["start_sec"]
            )

            delays.append(delay)

    false_alerts = [
        alert
        for index, alert in enumerate(alerts)
        if index not in matched_alerts
    ]

    actual_falls = len(fall_events)
    missed_falls = actual_falls - detected_count

    recall = (
        detected_count / actual_falls
        if actual_falls > 0
        else 0
    )

    print("\nEVENT-LEVEL RESULTS")
    print("-" * 50)

    print("Actual fall events:", actual_falls)
    print("Detected fall events:", detected_count)
    print("Missed fall events:", missed_falls)
    print("False alert episodes:", len(false_alerts))
    print(f"Fall event recall: {recall:.2%}")

    if delays:
        print(
            "Average detection delay:",
            f"{np.mean(delays):.2f} seconds",
        )
    else:
        print("Average detection delay: N/A")

    return false_alerts


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels, videos = training.load_labels()

    all_predictions = []
    all_alerts = []

    print("\nRULE-BASED TEMPORAL FALL EVALUATION")
    print("=" * 50)

    for video_name in sorted(labels["video"].unique()):
        video_path = videos[video_name]

        print(f"\nProcessing: {video_name}")

        predictions, alerts = replay_video(
            video_path
        )

        all_predictions.extend(predictions)
        all_alerts.extend(alerts)

        print(
            f"  Potential alerts: {len(alerts)}"
        )

    false_alerts = evaluate_events(
        labels,
        all_alerts,
    )

    # ----------------------------------------
    # Save the same three evaluation CSVs.
    # ----------------------------------------

    pd.DataFrame(
        all_predictions,
        columns=[
            "video",
            "timestamp_sec",
            "fall_probability",
            "predicted_fall",
            "fall_state",
            "posture",
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
            "fall_state",
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
            "fall_state",
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
