import csv
import os
import pytest

from src.yolo import YOLODetector


IMAGE_FOLDER = "data/test"
GROUND_TRUTH_FILE = "ground_truth.csv"

CONFIDENCE_LEVELS = [0.5, 0.4, 0.3, 0.2, 0.1]


def load_ground_truth():
    ground_truth = {}

    with open(GROUND_TRUTH_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            ground_truth[row["image"]] = row["condition"]

    return ground_truth


def get_image_files():
    image_files = []

    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(filename)

    image_files.sort()

    return image_files


def test_yolo_detection_confidence():

    ground_truth = load_ground_truth()
    image_files = get_image_files()

    assert len(image_files) == 20
    assert len(ground_truth) == 20

    print()
    print("YOLO detection confidence experiment")
    print("=" * 60)

    for confidence in CONFIDENCE_LEVELS:

        detector = YOLODetector(confidence=confidence)

        detected_count = 0
        missed_count = 0

        print()
        print(f"Confidence threshold: {confidence}")
        print("-" * 60)

        for image_name in image_files:

            image_path = os.path.join(
                IMAGE_FOLDER,
                image_name
            )

            persons = detector.detect_persons(image_path)

            if len(persons) > 0:
                detected_count += 1

                print(
                    f"{image_name:<20} "
                    f"DETECTED ({len(persons)} person)"
                )

            else:
                missed_count += 1

                print(
                    f"{image_name:<20} "
                    f"NO PERSON DETECTED"
                )

        print()
        print(
            f"Detected: {detected_count}/20 | "
            f"Missed: {missed_count}/20"
        )

    print()
    print("=" * 60)