import csv
import os
import cv2

from src.yolo import YOLODetector
from src.yolo_fall_detection import (
    calculate_bbox_ratio,
    classify_fall
)


IMAGE_FOLDER = "data/test"
GROUND_TRUTH_FILE = "ground_truth.csv"
RESULTS_FOLDER = "results"

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
    """Load image ground-truth labels from CSV."""

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
    """Get all test image filenames."""

    image_files = sorted(
        [
            file_name
            for file_name in os.listdir(IMAGE_FOLDER)
            if file_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]
    )

    return image_files


def get_highest_confidence_person(persons):
    """
    Select the person detection with the
    highest confidence.
    """

    if not persons:
        return None

    return max(
        persons,
        key=lambda person: person["confidence"]
    )


def test_yolo_on_test_images():

    detector = YOLODetector()

    ground_truth = load_ground_truth()
    image_files = get_image_files()

    assert len(image_files) == 20, (
        f"Expected 20 test images, found {len(image_files)}."
    )

    assert len(ground_truth) == 20, (
        f"Expected 20 ground-truth labels, "
        f"found {len(ground_truth)}."
    )

    os.makedirs(
        RESULTS_FOLDER,
        exist_ok=True
    )

    raw_results = []

    for file_name in image_files:

        image_path = os.path.join(
            IMAGE_FOLDER,
            file_name
        )

        image = cv2.imread(image_path)

        assert image is not None, (
            f"Could not load {file_name}."
        )

        persons = detector.detect_persons(image)

        best_person = get_highest_confidence_person(
            persons
        )

        if best_person is None:

            raw_results.append({
                "image": file_name,
                "ground_truth": ground_truth[file_name],
                "persons_detected": 0,
                "confidence": "",
                "bbox": "",
                "ratio": ""
            })

        else:

            bbox = best_person["bbox"]

            ratio = calculate_bbox_ratio(bbox)

            raw_results.append({
                "image": file_name,
                "ground_truth": ground_truth[file_name],
                "persons_detected": len(persons),
                "confidence": (
                    f"{best_person['confidence']:.3f}"
                ),
                "bbox": str(bbox),
                "ratio": f"{ratio:.3f}"
            })

    raw_file = os.path.join(
        RESULTS_FOLDER,
        "yolo_raw_results.csv"
    )

    with open(
        raw_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "image",
            "ground_truth",
            "persons_detected",
            "confidence",
            "bbox",
            "ratio"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(raw_results)

    print()
    print("Raw YOLO results saved to:")
    print(raw_file)

    evaluate_thresholds(raw_results)


def evaluate_thresholds(raw_results):

    threshold_results = []

    for threshold in THRESHOLDS:

        correct = 0
        incorrect = 0

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for result in raw_results:

            actual_fall = (
                result["ground_truth"] == "falling"
            )

            if result["ratio"] == "":

                predicted_fall = False

            else:

                ratio = float(result["ratio"])

                predicted_fall = classify_fall(
                    ratio,
                    threshold
                )

            if predicted_fall == actual_fall:

                correct += 1

            else:

                incorrect += 1

            if actual_fall and predicted_fall:
                true_positive += 1

            elif not actual_fall and not predicted_fall:
                true_negative += 1

            elif not actual_fall and predicted_fall:
                false_positive += 1

            elif actual_fall and not predicted_fall:
                false_negative += 1

        accuracy = correct / len(raw_results)

        threshold_results.append({
            "threshold": threshold,
            "correct": correct,
            "incorrect": incorrect,
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "accuracy": f"{accuracy:.3f}"
        })

    results_file = os.path.join(
        RESULTS_FOLDER,
        "yolo_threshold_results.csv"
    )

    with open(
        results_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "threshold",
            "correct",
            "incorrect",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
            "accuracy"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(threshold_results)

    print()
    print("Threshold evaluation saved to:")
    print(results_file)

    print()
    print("Threshold results:")

    for result in threshold_results:

        print(
            f"Threshold {result['threshold']:.1f} | "
            f"Accuracy {result['accuracy']} | "
            f"TP {result['true_positive']} | "
            f"TN {result['true_negative']} | "
            f"FP {result['false_positive']} | "
            f"FN {result['false_negative']}"
        )