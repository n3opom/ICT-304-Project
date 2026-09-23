import csv
import os
import pytest

from src.yolo import YOLODetector
from src.yolo_fall_detection import calculate_bbox_ratio, classify_fall


IMAGE_FOLDER = "data/test"
GROUND_TRUTH_FILE = "ground_truth.csv"

RAW_RESULTS_FILE = "results/yolo_confidence_raw_results.csv"
IMAGE_RESULTS_FILE = "results/yolo_image_results.csv"
COMBINATION_RESULTS_FILE = "results/yolo_combination_results.csv"

CONFIDENCE_LEVELS = [0.5, 0.4, 0.3, 0.2, 0.1]

THRESHOLDS = [
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4
]


def load_ground_truth():
    ground_truth = {}

    with open(
        GROUND_TRUTH_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            ground_truth[row["image"]] = row["condition"]

    return ground_truth


def get_image_files():
    image_files = []

    for filename in os.listdir(IMAGE_FOLDER):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            image_files.append(filename)

    image_files.sort()

    return image_files


def calculate_metrics(
    true_positive,
    true_negative,
    false_positive,
    false_negative
):

    correct = true_positive + true_negative
    incorrect = false_positive + false_negative

    total = correct + incorrect

    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0.0

    if true_positive + false_positive > 0:
        precision = (
            true_positive /
            (true_positive + false_positive)
        )
    else:
        precision = 0.0

    if true_positive + false_negative > 0:
        recall = (
            true_positive /
            (true_positive + false_negative)
        )
    else:
        recall = 0.0

    if precision + recall > 0:
        f1_score = (
            2 * precision * recall /
            (precision + recall)
        )
    else:
        f1_score = 0.0

    return (
        correct,
        incorrect,
        accuracy,
        precision,
        recall,
        f1_score
    )


def test_yolo_confidence_and_thresholds():

    os.makedirs("results", exist_ok=True)

    ground_truth = load_ground_truth()
    image_files = get_image_files()

    assert len(image_files) == 20
    assert len(ground_truth) == 20

    print()
    print("=" * 80)
    print("YOLO CONFIDENCE + FALL THRESHOLD EXPERIMENT")
    print("=" * 80)

    all_raw_results = []

    # =========================================================
    # STEP 1
    # Run YOLO at each confidence level
    # =========================================================

    for confidence in CONFIDENCE_LEVELS:

        print()
        print(
            f"Running YOLO with confidence = "
            f"{confidence}"
        )

        detector = YOLODetector(
            confidence=confidence
        )

        for image_name in image_files:

            image_path = os.path.join(
                IMAGE_FOLDER,
                image_name
            )

            persons = detector.detect_persons(
                image_path
            )

            person_results = []

            for person in persons:

                ratio = calculate_bbox_ratio(
                    person["bbox"]
                )

                person_results.append({
                    "confidence": person["confidence"],
                    "ratio": ratio
                })

            all_raw_results.append({
                "confidence_threshold": confidence,
                "image": image_name,
                "condition": ground_truth[image_name],
                "persons": person_results
            })

    # =========================================================
    # STEP 2
    # Save raw YOLO detection results
    # =========================================================

    with open(
        RAW_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "confidence_threshold",
            "image",
            "condition",
            "person_count",
            "person_confidences",
            "bbox_ratios"
        ])

        for result in all_raw_results:

            confidences = ";".join(
                f"{person['confidence']:.4f}"
                for person in result["persons"]
            )

            ratios = ";".join(
                f"{person['ratio']:.4f}"
                for person in result["persons"]
            )

            writer.writerow([
                result["confidence_threshold"],
                result["image"],
                result["condition"],
                len(result["persons"]),
                confidences,
                ratios
            ])

    print()
    print(
        f"Raw detection results saved to: "
        f"{RAW_RESULTS_FILE}"
    )

    # =========================================================
    # STEP 3
    # Test every confidence × fall threshold combination
    # =========================================================

    combination_results = []
    image_results = []

    for confidence in CONFIDENCE_LEVELS:

        confidence_results = [
            result
            for result in all_raw_results
            if result["confidence_threshold"] == confidence
        ]

        for threshold in THRESHOLDS:

            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            for result in confidence_results:

                actual_fall = (
                    result["condition"] == "falling"
                )

                persons = result["persons"]

                # -------------------------------------------------
                # No person detected
                # -------------------------------------------------

                if len(persons) == 0:

                    predicted_fall = False
                    prediction_reason = "No person detected"

                else:

                    predicted_fall = False
                    prediction_reason = "No person exceeded threshold"

                    # -------------------------------------------------
                    # ANY detected person can trigger FALL
                    # -------------------------------------------------

                    for person in persons:

                        if classify_fall(
                            person["ratio"],
                            threshold
                        ):

                            predicted_fall = True

                            prediction_reason = (
                                "At least one person "
                                "exceeded threshold"
                            )

                            break

                # -------------------------------------------------
                # Determine classification
                # -------------------------------------------------

                if actual_fall and predicted_fall:

                    true_positive += 1
                    classification = "TP"

                elif not actual_fall and not predicted_fall:

                    true_negative += 1
                    classification = "TN"

                elif not actual_fall and predicted_fall:

                    false_positive += 1
                    classification = "FP"

                else:

                    false_negative += 1
                    classification = "FN"

                # -------------------------------------------------
                # Get bbox ratios for CSV
                # -------------------------------------------------

                ratios = ";".join(
                    f"{person['ratio']:.4f}"
                    for person in persons
                )

                # -------------------------------------------------
                # Save individual image result
                # -------------------------------------------------

                image_results.append({
                    "confidence": confidence,
                    "threshold": threshold,
                    "image": result["image"],
                    "actual": result["condition"],
                    "predicted": (
                        "FALL"
                        if predicted_fall
                        else "NOT FALL"
                    ),
                    "classification": classification,
                    "correct": (
                        "True"
                        if (
                            (actual_fall and predicted_fall)
                            or
                            (not actual_fall and not predicted_fall)
                        )
                        else "False"
                    ),
                    "person_count": len(persons),
                    "bbox_ratios": ratios,
                    "reason": prediction_reason
                })

            (
                correct,
                incorrect,
                accuracy,
                precision,
                recall,
                f1_score
            ) = calculate_metrics(
                true_positive,
                true_negative,
                false_positive,
                false_negative
            )

            combination_results.append({
                "confidence": confidence,
                "threshold": threshold,
                "correct": correct,
                "incorrect": incorrect,
                "true_positive": true_positive,
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            })

    # =========================================================
    # STEP 4
    # Save IMAGE-LEVEL results
    # =========================================================

    with open(
        IMAGE_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "confidence",
            "threshold",
            "image",
            "actual",
            "predicted",
            "classification",
            "correct",
            "person_count",
            "bbox_ratios",
            "reason"
        ])

        for result in image_results:

            writer.writerow([
                result["confidence"],
                result["threshold"],
                result["image"],
                result["actual"],
                result["predicted"],
                result["classification"],
                result["correct"],
                result["person_count"],
                result["bbox_ratios"],
                result["reason"]
            ])

    print()
    print(
        f"Image-level results saved to: "
        f"{IMAGE_RESULTS_FILE}"
    )

    # =========================================================
    # STEP 5
    # Save aggregate results
    # =========================================================

    with open(
        COMBINATION_RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "confidence",
            "threshold",
            "correct",
            "incorrect",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
            "accuracy",
            "precision",
            "recall",
            "f1_score"
        ])

        for result in combination_results:

            writer.writerow([
                result["confidence"],
                result["threshold"],
                result["correct"],
                result["incorrect"],
                result["true_positive"],
                result["true_negative"],
                result["false_positive"],
                result["false_negative"],
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1_score']:.4f}"
            ])

    print(
        f"Combination results saved to: "
        f"{COMBINATION_RESULTS_FILE}"
    )

    # =========================================================
    # STEP 6
    # Print aggregate results
    # =========================================================

    print()
    print("=" * 100)
    print("ALL CONFIDENCE × FALL THRESHOLD RESULTS")
    print("=" * 100)

    for result in combination_results:

        print(
            f"Confidence {result['confidence']:.1f} | "
            f"Threshold {result['threshold']:.1f} | "
            f"Accuracy {result['accuracy']:.3f} | "
            f"Precision {result['precision']:.3f} | "
            f"Recall {result['recall']:.3f} | "
            f"F1 {result['f1_score']:.3f} | "
            f"TP {result['true_positive']} | "
            f"TN {result['true_negative']} | "
            f"FP {result['false_positive']} | "
            f"FN {result['false_negative']}"
        )

    # =========================================================
    # STEP 7
    # Print misclassified images
    # =========================================================

    print()
    print("=" * 100)
    print("MISCLASSIFIED IMAGES")
    print("=" * 100)

    misclassified_count = 0

    for result in image_results:

        if result["correct"] == "False":

            misclassified_count += 1

            print(
                f"Confidence {result['confidence']:.1f} | "
                f"Threshold {result['threshold']:.1f} | "
                f"Image {result['image']} | "
                f"Actual {result['actual']} | "
                f"Predicted {result['predicted']} | "
                f"{result['classification']} | "
                f"Ratios [{result['bbox_ratios']}] | "
                f"{result['reason']}"
            )

    print()
    print(
        f"Total misclassified image/parameter cases: "
        f"{misclassified_count}"
    )

    # There should be 40 combinations × 20 images
    assert len(image_results) == 800

    assert len(combination_results) == 40