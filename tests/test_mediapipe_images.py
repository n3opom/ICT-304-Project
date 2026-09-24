from pathlib import Path
import csv
import math

import cv2
import pytest
from mediapipe.tasks.python.vision import PoseLandmark

from src.mediapipe import MediaPipePoseDetector
from src.mediapipe_fall_detection import MediaPipeFallDetector


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "test_images"
GROUND_TRUTH_FILE = PROJECT_ROOT / "ground_truth.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_RESULTS_FILE = RESULTS_DIR / "mediapipe_raw_results.csv"
IMAGE_RESULTS_FILE = RESULTS_DIR / "mediapipe_image_results.csv"
COMBINATION_RESULTS_FILE = RESULTS_DIR / "mediapipe_combination_results.csv"


# ============================================================
# Experiment settings
# ============================================================

DETECTION_CONFIDENCES = [0.3, 0.4, 0.5, 0.6, 0.7]

TORSO_ANGLE_THRESHOLDS = [
    30, 35, 40, 45, 50, 55, 60, 65
]

MIN_LANDMARK_VISIBILITY = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1


# ============================================================
# Ground-truth helpers
# ============================================================

def condition_to_label(condition):
    """
    Convert the condition in ground_truth.csv into the binary
    classification used by the experiment.
    """

    condition = condition.strip().lower()

    if condition in {"fall", "falling"}:
        return "FALL"

    if condition in {
        "no_fall",
        "no fall",
        "not fallen",
        "sitting",
        "standing"
    }:
        return "NO_FALL"

    raise ValueError(
        f"Unknown ground-truth condition: '{condition}'"
    )


def load_ground_truth():
    """
    Read ground_truth.csv.

    Expected columns:
        image
        condition

    The image filename is used as the key, so filenames containing
    spaces are handled correctly.
    """

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found:\n{GROUND_TRUTH_FILE}"
        )

    ground_truth = {}

    with open(GROUND_TRUTH_FILE, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("ground_truth.csv has no header.")

        required_columns = {"image", "condition"}

        if not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(
                "ground_truth.csv must contain columns: image, condition"
            )

        for row in reader:
            image_name = row["image"].strip()

            if not image_name:
                continue

            if image_name in ground_truth:
                raise ValueError(
                    f"Duplicate image in ground_truth.csv: {image_name}"
                )

            ground_truth[image_name] = {
                "condition": row["condition"].strip(),
                "actual_label": condition_to_label(row["condition"])
            }

    return ground_truth


# ============================================================
# Image discovery
# ============================================================

def get_test_images():
    """
    Automatically iterate through every supported image in
    data/test_images.

    The code does NOT assume image01.jpg, image02.jpg, etc.
    """

    if not TEST_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n{TEST_IMAGE_DIR}"
        )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    images = [
        path
        for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    ]

    return sorted(images, key=lambda path: path.name.lower())


# ============================================================
# Pose / angle helpers
# ============================================================

def get_landmark_visibility(landmarks, landmark_index):
    """
    Get visibility for a landmark.

    Returns an empty value if the landmark does not provide
    a visibility value.
    """

    landmark = landmarks[landmark_index]

    visibility = getattr(landmark, "visibility", None)

    if visibility is None:
        return ""

    return float(visibility)


def calculate_torso_angle_and_reason(landmarks):
    """
    Calculate the torso angle using:

        left shoulder
        right shoulder
        left hip
        right hip

    Returns:
        (angle, reason)

    angle is None when the required landmarks cannot be used.
    """

    if landmarks is None:
        return None, "no person detected"

    left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[PoseLandmark.RIGHT_HIP.value]

    required = [
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip
    ]

    for landmark in required:
        if landmark is None:
            return None, "required torso landmarks unavailable"

        visibility = getattr(landmark, "visibility", None)

        if visibility is not None and visibility < MIN_LANDMARK_VISIBILITY:
            return None, "required torso landmark visibility below minimum"

    shoulder_x = (
        left_shoulder.x + right_shoulder.x
    ) / 2.0

    shoulder_y = (
        left_shoulder.y + right_shoulder.y
    ) / 2.0

    hip_x = (
        left_hip.x + right_hip.x
    ) / 2.0

    hip_y = (
        left_hip.y + right_hip.y
    ) / 2.0

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y

    if dx == 0 and dy == 0:
        return None, "torso points are identical"

    angle = math.degrees(math.atan2(abs(dy), abs(dx)))

    return angle, ""


# ============================================================
# Classification helpers
# ============================================================

def calculate_prediction(torso_angle, threshold, person_detected):
    """
    Apply the torso-angle threshold.

    A smaller torso angle means the torso is closer to horizontal,
    so it is classified as FALL.
    """

    if not person_detected:
        return "NO_FALL"

    if torso_angle is None:
        return "NO_FALL"

    if torso_angle < threshold:
        return "FALL"

    return "NO_FALL"


def calculate_confusion(actual, predicted):
    tp = int(actual == "FALL" and predicted == "FALL")
    tn = int(actual == "NO_FALL" and predicted == "NO_FALL")
    fp = int(actual == "NO_FALL" and predicted == "FALL")
    fn = int(actual == "FALL" and predicted == "NO_FALL")

    return tp, tn, fp, fn


def calculate_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return accuracy, precision, recall, f1


# ============================================================
# CSV writers
# ============================================================

def write_csv(path, fieldnames, rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main experiment
# ============================================================

def run_mediapipe_experiment():
    ground_truth = load_ground_truth()
    image_paths = get_test_images()

    if len(image_paths) != 20:
        raise AssertionError(
            f"Expected 20 test images, but found {len(image_paths)} "
            f"in {TEST_IMAGE_DIR}"
        )

    image_names = {path.name for path in image_paths}
    ground_truth_names = set(ground_truth.keys())

    missing_ground_truth = image_names - ground_truth_names
    missing_images = ground_truth_names - image_names

    if missing_ground_truth:
        raise AssertionError(
            "These test images are missing from ground_truth.csv:\n"
            + "\n".join(sorted(missing_ground_truth))
        )

    if missing_images:
        raise AssertionError(
            "These ground-truth images are missing from data/test_images:\n"
            + "\n".join(sorted(missing_images))
        )

    raw_rows = []
    image_rows = []
    combination_rows = []

    # Cache raw MediaPipe results so the same image is not processed
    # repeatedly for every torso-angle threshold.
    cached_results = {}

    # --------------------------------------------------------
    # Run MediaPipe for each detection confidence.
    # --------------------------------------------------------

    for detection_confidence in DETECTION_CONFIDENCES:

        print(
            f"\nDetection confidence: {detection_confidence}"
        )

        pose_detector = MediaPipePoseDetector(
            min_detection_confidence=detection_confidence,
            min_pose_presence_confidence=detection_confidence,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        try:
            for image_path in image_paths:

                image = cv2.imread(str(image_path))

                if image is None:
                    raise ValueError(
                        f"Could not read image: {image_path}"
                    )

                landmarks = pose_detector.get_landmarks(image)

                person_detected = landmarks is not None

                torso_angle, reason = (
                    calculate_torso_angle_and_reason(landmarks)
                )

                left_shoulder_visibility = ""
                right_shoulder_visibility = ""
                left_hip_visibility = ""
                right_hip_visibility = ""

                if landmarks is not None:
                    left_shoulder_visibility = get_landmark_visibility(
                        landmarks,
                        PoseLandmark.LEFT_SHOULDER.value
                    )

                    right_shoulder_visibility = get_landmark_visibility(
                        landmarks,
                        PoseLandmark.RIGHT_SHOULDER.value
                    )

                    left_hip_visibility = get_landmark_visibility(
                        landmarks,
                        PoseLandmark.LEFT_HIP.value
                    )

                    right_hip_visibility = get_landmark_visibility(
                        landmarks,
                        PoseLandmark.RIGHT_HIP.value
                    )

                cached_results[
                    (detection_confidence, image_path.name)
                ] = {
                    "person_detected": person_detected,
                    "torso_angle": torso_angle,
                    "reason": reason,
                    "left_shoulder_visibility": left_shoulder_visibility,
                    "right_shoulder_visibility": right_shoulder_visibility,
                    "left_hip_visibility": left_hip_visibility,
                    "right_hip_visibility": right_hip_visibility
                }

                # Raw results contain one row per image per
                # detection-confidence setting.
                raw_rows.append({
                    "image": image_path.name,
                    "detection_confidence": detection_confidence,
                    "person_detected": person_detected,
                    "torso_angle": (
                        ""
                        if torso_angle is None
                        else round(torso_angle, 4)
                    ),
                    "left_shoulder_visibility":
                        left_shoulder_visibility,
                    "right_shoulder_visibility":
                        right_shoulder_visibility,
                    "left_hip_visibility":
                        left_hip_visibility,
                    "right_hip_visibility":
                        right_hip_visibility,
                    "reason": reason
                })

        finally:
            pose_detector.close()

    # --------------------------------------------------------
    # Apply every torso-angle threshold.
    # --------------------------------------------------------

    for detection_confidence in DETECTION_CONFIDENCES:

        for torso_angle_threshold in TORSO_ANGLE_THRESHOLDS:

            tp = 0
            tn = 0
            fp = 0
            fn = 0

            for image_path in image_paths:

                image_name = image_path.name
                actual_label = ground_truth[
                    image_name
                ]["actual_label"]

                cached = cached_results[
                    (detection_confidence, image_name)
                ]

                predicted_label = calculate_prediction(
                    cached["torso_angle"],
                    torso_angle_threshold,
                    cached["person_detected"]
                )

                current_tp, current_tn, current_fp, current_fn = (
                    calculate_confusion(
                        actual_label,
                        predicted_label
                    )
                )

                tp += current_tp
                tn += current_tn
                fp += current_fp
                fn += current_fn

                image_rows.append({
                    "image": image_name,
                    "condition": ground_truth[
                        image_name
                    ]["condition"],
                    "actual_label": actual_label,
                    "detection_confidence": detection_confidence,
                    "torso_angle_threshold":
                        torso_angle_threshold,
                    "predicted_label": predicted_label,
                    "TP": current_tp,
                    "TN": current_tn,
                    "FP": current_fp,
                    "FN": current_fn,
                    "correct":
                        int(actual_label == predicted_label),
                    "person_detected":
                        cached["person_detected"],
                    "torso_angle": (
                        ""
                        if cached["torso_angle"] is None
                        else round(cached["torso_angle"], 4)
                    ),
                    "left_shoulder_visibility":
                        cached["left_shoulder_visibility"],
                    "right_shoulder_visibility":
                        cached["right_shoulder_visibility"],
                    "left_hip_visibility":
                        cached["left_hip_visibility"],
                    "right_hip_visibility":
                        cached["right_hip_visibility"],
                    "reason": cached["reason"]
                })

            accuracy, precision, recall, f1 = calculate_metrics(
                tp,
                tn,
                fp,
                fn
            )

            combination_rows.append({
                "detection_confidence":
                    detection_confidence,
                "torso_angle_threshold":
                    torso_angle_threshold,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "accuracy": round(accuracy, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "F1": round(f1, 6)
            })

    # ========================================================
    # Write the three agreed CSV files
    # ========================================================

    write_csv(
        RAW_RESULTS_FILE,
        [
            "image",
            "detection_confidence",
            "person_detected",
            "torso_angle",
            "left_shoulder_visibility",
            "right_shoulder_visibility",
            "left_hip_visibility",
            "right_hip_visibility",
            "reason"
        ],
        raw_rows
    )

    write_csv(
        IMAGE_RESULTS_FILE,
        [
            "image",
            "condition",
            "actual_label",
            "detection_confidence",
            "torso_angle_threshold",
            "predicted_label",
            "TP",
            "TN",
            "FP",
            "FN",
            "correct",
            "person_detected",
            "torso_angle",
            "left_shoulder_visibility",
            "right_shoulder_visibility",
            "left_hip_visibility",
            "right_hip_visibility",
            "reason"
        ],
        image_rows
    )

    write_csv(
        COMBINATION_RESULTS_FILE,
        [
            "detection_confidence",
            "torso_angle_threshold",
            "TP",
            "TN",
            "FP",
            "FN",
            "accuracy",
            "precision",
            "recall",
            "F1"
        ],
        combination_rows
    )

    # ========================================================
    # Validation
    # ========================================================

    assert len(raw_rows) == 20 * 5
    assert len(image_rows) == 20 * 5 * 8
    assert len(combination_rows) == 5 * 8

    print("\nMediaPipe experiment completed.")
    print(f"Raw results:         {RAW_RESULTS_FILE}")
    print(f"Image results:       {IMAGE_RESULTS_FILE}")
    print(f"Combination results: {COMBINATION_RESULTS_FILE}")

    print(f"\nRaw rows:         {len(raw_rows)}")
    print(f"Image rows:       {len(image_rows)}")
    print(f"Combination rows: {len(combination_rows)}")


# ============================================================
# Pytest entry point
# ============================================================

def test_mediapipe_image_experiment():
    run_mediapipe_experiment()


if __name__ == "__main__":
    run_mediapipe_experiment()
