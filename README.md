# ICT-304 Project — YOLO Branch

## Overview

The `feature/yolo` branch develops the **YOLO bounding-box** approach for the ICT-304 AI System Design fall-detection project.

The branch uses a YOLO model to detect objects in an image, identifies detections belonging to the `person` class, and then uses the person's bounding-box width-to-height ratio as the fall-detection feature.

The YOLO detector itself performs **object detection**. The separate fall-detection component interprets the bounding-box geometry to classify whether a detected person is considered to be falling.

## Branch Structure

```text
ICT-304-Project/
│
├── .vscode/
│
├── data/
│   └── test/
│       └── [test images]
│
├── results/
│   ├── yolo_combination_results.csv
│   ├── yolo_confidence_raw_results.csv
│   ├── yolo_image_results.csv
│   ├── yolo_raw_results.csv
│   └── yolo_threshold_results.csv
│
├── src/
│   ├── yolo.py
│   ├── yolo_detection.py
│   └── yolo_fall_detection.py
│
├── tests/
│   ├── test_yolo.py
│   └── test_yolo_images.py
│
├── .gitignore
├── README.md
├── ground_truth.csv
├── requirements.txt
└── yolo11n.pt
```

## YOLO Model

The branch includes:

```text
yolo11n.pt
```

This is the YOLO model used by the detector.

The detector wrapper is implemented in:

```text
src/yolo.py
```

and uses the Ultralytics YOLO interface.

The detector accepts a model path and a confidence value:

```python
YOLODetector(
    model_path="yolo11n.pt",
    confidence=0.5
)
```

The default confidence value in the detector is `0.5`. fileciteturn4file0L4-L9

## YOLO Detection

The detector receives an OpenCV image/frame and runs YOLO inference using the configured confidence threshold.

For each detected object, it records:

- Class ID
- Class name
- Detection confidence
- Bounding-box coordinates

The bounding box is represented as:

```text
(x1, y1, x2, y2)
```

where the coordinates describe the top-left and bottom-right corners of the detected object. fileciteturn4file0L11-L19 fileciteturn4file0L38-L55

The detector also provides a method for filtering the detections so that only objects classified as `person` are retained. fileciteturn4file1L61-L75

## Bounding-Box Fall Detection

The fall-detection logic is implemented in:

```text
src/yolo_fall_detection.py
```

The system calculates the width-to-height ratio of each detected person's bounding box:

```text
ratio = width / height
```

The width and height are calculated from the bounding-box coordinates:

```text
width  = x2 - x1
height = y2 - y1
```

If the height is zero or negative, the function returns `0.0`. fileciteturn4file2L1-L15

### Classification Rule

The fall classifier compares the bounding-box ratio with a supplied threshold:

```text
ratio >= threshold  → FALL
ratio < threshold   → NOT FALL
```

The classification function therefore uses the bounding box's shape as the fall indicator rather than body-pose landmarks. fileciteturn4file2L18-L24

Conceptually:

```text
              Bounding Box
          ┌──────────────────┐
          │                  │
          │      Person      │
          │                  │
          └──────────────────┘
              width / height
                    │
                    ▼
             Compare Ratio
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    ratio >= threshold   ratio < threshold
          │                   │
          ▼                   ▼
        FALL              NOT FALL
```

## Test Dataset

The test images are stored in:

```text
data/test/
```

The ground-truth labels are stored in:

```text
ground_truth.csv
```

The image-testing script loads the ground-truth file and maps each image filename to its expected condition. fileciteturn4file4L30-L45

The test procedure expects:

```text
20 test images
20 ground-truth entries
```

and explicitly asserts both counts. fileciteturn4file4L114-L122

## Parameter Testing

The branch tests two parameters:

### 1. YOLO confidence threshold

The following five confidence values are tested:

```text
0.5
0.4
0.3
0.2
0.1
```

### 2. Fall-detection bounding-box threshold

The following eight thresholds are tested:

```text
0.7
0.8
0.9
1.0
1.1
1.2
1.3
1.4
```

This produces:

```text
5 × 8 = 40 parameter combinations
```

The parameter values are defined directly in `test_yolo_images.py`. fileciteturn4file4L9-L18

For each confidence level, YOLO is run across all 20 test images. The detected people and their bounding-box ratios are then stored before evaluating every confidence × fall-threshold combination. fileciteturn4file4L129-L176 fileciteturn4file4L229-L285

## Multi-Person Handling

The testing implementation does not assume that an image contains only one detected person.

For every image, all detected `person` objects are collected and their bounding-box ratios are calculated. fileciteturn4file4L155-L176

During classification, **any detected person exceeding the fall threshold can cause the image to be classified as `FALL`**:

```text
If at least one detected person
has ratio >= threshold
        │
        ▼
      FALL
```

This behaviour is implemented explicitly in the image-testing code. fileciteturn4file4L260-L292

## Result Files

The branch generates several CSV files in the `results/` directory.

### `yolo_confidence_raw_results.csv`

Contains raw YOLO detections generated across the tested confidence levels.

The stored information includes:

- Confidence threshold
- Image
- Ground-truth condition
- Number of detected persons
- Person detection confidences
- Bounding-box ratios

The testing code writes these fields to the raw-results CSV. fileciteturn4file4L179-L220

### `yolo_image_results.csv`

Contains the result for each individual image and parameter combination.

The output includes:

- Confidence
- Fall threshold
- Image
- Actual condition
- Predicted result
- TP/TN/FP/FN classification
- Correct/incorrect result
- Person count
- Bounding-box ratios
- Prediction reason

These fields are written by the image-level results stage. fileciteturn4file4L385-L424

### `yolo_combination_results.csv`

Contains the aggregate performance for each confidence × threshold combination.

The recorded metrics include:

- Correct
- Incorrect
- True Positive
- True Negative
- False Positive
- False Negative
- Accuracy
- Precision
- Recall
- F1 score

fileciteturn4file4L433-L476

### `yolo_raw_results.csv` and `yolo_threshold_results.csv`

These are also present in the branch's `results/` directory and represent additional generated result outputs from the YOLO testing/evaluation workflow.

Their exact generation logic should be interpreted from the corresponding historical test scripts/results rather than assumed to have identical schemas to the current combination and image-level result files.

## Evaluation Metrics

The testing code calculates:

### Accuracy

```text
(TP + TN) / (TP + TN + FP + FN)
```

### Precision

```text
TP / (TP + FP)
```

### Recall

```text
TP / (TP + FN)
```

### F1 Score

```text
2 × Precision × Recall
----------------------
   Precision + Recall
```

The implementation explicitly calculates these four metrics after each confidence × threshold combination. fileciteturn4file4L63-L110

## Measured Experiment Size

The testing code verifies that the image-level experiment contains:

```text
20 images × 40 parameter combinations = 800 cases
```

and verifies that 40 aggregate parameter combinations are produced. fileciteturn4file4L544-L547

The supplied result files contain:

```text
Combination results: 40 rows
Image-level results: 800 rows
Ground-truth entries: 20 rows
```

The tested confidence values and thresholds in the supplied combination results are:

```text
Confidence values: [0.1, 0.2, 0.3, 0.4, 0.5]
Threshold values: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
```

## Classification and Error Analysis

For every image/parameter combination, the testing script assigns one of:

```text
TP  = True Positive
TN  = True Negative
FP  = False Positive
FN  = False Negative
```

It also records whether the individual prediction was correct and provides a reason for the prediction. fileciteturn4file4L294-L353

The script additionally prints the misclassified image/parameter cases, including:

- Confidence
- Threshold
- Image
- Actual condition
- Predicted condition
- Classification
- Bounding-box ratios
- Prediction reason

This provides a way to identify which images were misclassified under particular parameter combinations. fileciteturn4file4L509-L541

## Unit Tests

The basic YOLO tests are located in:

```text
tests/test_yolo.py
```

The current tests verify that:

1. The YOLO model can be loaded.
2. Passing `None` as an input frame raises a `ValueError`.
3. The detector accepts and stores a specified confidence value.

These tests are implemented in `test_yolo.py`. fileciteturn4file3L4-L23

The larger image-based experiment is located in:

```text
tests/test_yolo_images.py
```

This test performs the confidence × fall-threshold experiment and generates the CSV results.

## Installation

Create and activate a virtual environment if required:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install the branch dependencies:

```bash
pip install -r requirements.txt
```

The branch requirements are:

```text
ultralytics
opencv-python
pytest
```

The YOLO model file is included in the repository:

```text
yolo11n.pt
```

## Running the Tests

From the project root:

```bash
pytest -v
```

To run the basic YOLO unit tests:

```bash
pytest tests/test_yolo.py -v
```

To run the image-based parameter experiment:

```bash
pytest tests/test_yolo_images.py -v
```

The image experiment creates the `results/` directory when required and writes the generated CSV outputs there. fileciteturn4file4L114-L118

## YOLO Processing Pipeline

The branch processing flow can be represented as:

```text
Input Image
     │
     ▼
YOLO Object Detection
     │
     ▼
Detect Person Bounding Box
     │
     ▼
Calculate Width / Height Ratio
     │
     ▼
Compare Against Threshold
     │
     ├── Ratio >= Threshold
     │       │
     │       ▼
     │      FALL
     │
     └── Ratio < Threshold
             │
             ▼
          NOT FALL
```

For images containing multiple detected people, the current evaluation logic treats the image as a fall if at least one detected person exceeds the configured ratio threshold.

## Role in the Overall AI System

The YOLO branch provides the **bounding-box-based AI fall-detection technique** for comparison with the MediaPipe Pose approach.

Its processing is based on:

```text
Object Detection
      ↓
Person Bounding Box
      ↓
Bounding-Box Geometry
      ↓
Fall Classification
```

Unlike the MediaPipe branch, which uses body landmarks and torso orientation, this branch uses the geometry of the detected person's bounding box.

## Limitations of the Current Approach

- YOLO detects the person but does not inherently classify the person as falling; the fall decision is produced by the separate bounding-box ratio rule.
- The fall decision depends on a manually tested width-to-height threshold.
- The bounding-box ratio does not directly describe the person's actual body pose.
- Different camera angles, distances, occlusion, and body positions can change the bounding-box shape.
- The current evaluation treats an image as `FALL` if **any** detected person exceeds the threshold.
- The measured results are based on the supplied 20-image dataset and should not be interpreted as a general real-world performance guarantee.

## Summary

The `feature/yolo` branch implements a YOLO-based bounding-box approach for fall detection.

The branch:

1. Loads the YOLO model.
2. Detects objects in an input image.
3. Filters detections for the `person` class.
4. Extracts each person's bounding box.
5. Calculates the bounding-box width-to-height ratio.
6. Applies a configurable fall threshold.
7. Evaluates multiple confidence and threshold combinations.
8. Produces image-level and aggregate performance results.
9. Records TP, TN, FP, FN, accuracy, precision, recall, and F1 score.
10. Provides unit tests for the YOLO detector and an image-based evaluation test.

This branch provides the YOLO bounding-box implementation used for comparison with the MediaPipe Pose technique before selecting the approach for system integration.
