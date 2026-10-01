# ICT-304 Project — MediaPipe Branch

## Overview

The `feature/mediapipe` branch develops the **MediaPipe Pose** fall-detection technique for the ICT-304 AI System Design project.

This branch separates the MediaPipe implementation into two main components:

1. **Pose detection** — detects a human pose and obtains body landmarks using the MediaPipe Pose Landmarker.
2. **Fall detection** — uses the orientation of the person's torso to classify the pose as `FALL` or `NO_FALL`.

The branch also contains a parameter-testing framework that evaluates different MediaPipe detection-confidence values and torso-angle thresholds against a labelled test dataset.

## Branch Structure

```text
ICT-304-Project/
│
├── .vscode/
│
├── data/
│   └── test_images/
│       └── [test images]
│
├── results/
│   ├── mediapipe_combination_results.csv
│   ├── mediapipe_image_results.csv
│   └── mediapipe_raw_results.csv
│
├── src/
│   ├── mediapipe.py
│   └── mediapipe_fall_detection.py
│
├── tests/
│   ├── test_mediapipe.py
│   └── test_mediapipe_images.py
│
├── .gitignore
├── README.md
├── ground_truth.csv
├── pose_landmarker_full.task
└── requirements.txt
```

## MediaPipe Pose Detection

The main pose detector is implemented in:

```text
src/mediapipe.py
```

The branch uses the MediaPipe Tasks API and the `pose_landmarker_full.task` model.

The model file is located at the project root:

```text
pose_landmarker_full.task
```

The detector converts OpenCV BGR images to RGB before passing them to the MediaPipe Pose Landmarker. It then obtains the detected pose landmarks. fileciteturn1file1L69-L92

### Detection Parameters

The `MediaPipePoseDetector` supports the following parameters:

```python
MediaPipePoseDetector(
    model_path=None,
    min_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    num_poses=1
)
```

The detection-confidence parameters control the MediaPipe pose detection, pose presence, and tracking confidence thresholds.

The `num_poses` parameter controls the number of poses that MediaPipe is configured to detect. The current default is:

```text
num_poses = 1
```

Although the detector can be configured for more than one pose, `get_landmarks()` currently returns the landmarks belonging to the **first detected pose**. fileciteturn1file1L17-L23 fileciteturn1file1L96-L110

## Fall Detection Method

The fall-classification logic is implemented in:

```text
src/mediapipe_fall_detection.py
```

The system determines whether a detected pose represents a fall based on **torso orientation**.

The torso angle is defined approximately as:

```text
0°  = horizontal torso
90° = vertical torso
```

The detector uses the following four landmarks:

- Left shoulder
- Right shoulder
- Left hip
- Right hip

The midpoint of the two shoulders and the midpoint of the two hips are calculated. The vector between these two midpoints is then used to calculate the torso angle relative to the horizontal axis. fileciteturn1file0L59-L71 fileciteturn1file0L75-L135

### Fall Classification

The default torso-angle threshold is:

```text
60°
```

The classification rule is:

```text
torso angle < threshold  → FALL
torso angle ≥ threshold  → NO_FALL
```

If no pose is detected, the classifier returns:

```text
NO_PERSON
```

The fall detector also checks the visibility of the required landmarks before calculating the torso angle. fileciteturn1file0L18-L28 fileciteturn1file0L137-L174

## Test Dataset

The test images are stored in:

```text
data/test_images/
```

The ground-truth labels are stored separately in:

```text
ground_truth.csv
```

The ground-truth dataset contains 20 labelled images:

- 10 falling images
- 10 non-falling images

The falling images are labelled with:

```text
fall001.jpg
...
fall010.jpg
```

The non-falling images include sitting and standing conditions, such as:

```text
not fallen001.jpg
...
not fallen010.jpg
```

The ground-truth file uses the `condition` column to identify the expected condition, including `falling`, `sitting`, and `standing`.

The ground-truth labels are used to calculate classification performance during the parameter-testing process.

## Parameter Testing

The MediaPipe branch tests combinations of two parameters:

### Detection confidence

Five values are tested:

```text
0.3
0.4
0.5
0.6
0.7
```

### Torso-angle threshold

Eight values are tested:

```text
30°
35°
40°
45°
50°
55°
60°
65°
```

This produces:

```text
5 × 8 = 40 parameter combinations
```

Each combination is evaluated against the 20-image test dataset.

The resulting performance measures include:

- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
- Accuracy
- Precision
- Recall
- F1 score

## Results Files

The `results` directory contains three CSV outputs.

### 1. `mediapipe_raw_results.csv`

Contains the raw detection information for each image.

The recorded information includes:

- Image filename
- Detection confidence
- Whether a person was detected
- Torso angle
- Left shoulder visibility
- Right shoulder visibility
- Left hip visibility
- Right hip visibility
- Reason when detection could not be performed

This file is useful for examining the actual pose-detection output before applying the fall-classification threshold.

### 2. `mediapipe_image_results.csv`

Contains image-level classification results for each parameter combination.

The results include:

- Image
- Ground-truth condition
- Actual label
- Detection confidence
- Torso-angle threshold
- Predicted label
- TP/TN/FP/FN
- Correct/incorrect result
- Person detection status
- Torso angle
- Landmark visibility values
- Reason for the result

There are 800 rows in this result file, representing:

```text
20 images × 40 parameter combinations
```

### 3. `mediapipe_combination_results.csv`

Contains the aggregated performance for each of the 40 parameter combinations.

Each row represents one:

```text
Detection confidence × Torso-angle threshold
```

combination.

The output contains:

```text
detection_confidence
torso_angle_threshold
TP
TN
FP
FN
accuracy
precision
recall
F1
```

## Observed Parameter-Test Results

Based on the generated combination results, several parameter combinations produced:

```text
Accuracy  = 0.90
Precision = 1.00
Recall    = 0.80
F1        = 0.888889
```

For example, detection confidence values of `0.3`, `0.4`, and `0.5` produced these results at several torso-angle thresholds between `40°` and `60°`.

The corresponding confusion-matrix counts were:

```text
TP = 8
TN = 10
FP = 0
FN = 2
```

These results are based on the 20-image test dataset and should therefore be treated as the measured performance of this test set rather than a general performance guarantee.

## Example of Torso-Angle Interpretation

For a representative parameter combination of:

```text
Detection confidence = 0.5
Torso-angle threshold = 60°
```

a detected torso angle such as:

```text
13.3°
```

is classified as:

```text
FALL
```

because it is below the 60° threshold.

A torso angle such as:

```text
88.4°
```

is classified as:

```text
NO_FALL
```

because it is above the threshold.

This illustrates how the system uses body orientation rather than a bounding-box shape to determine the fall classification.

## Testing

The automated tests are located in:

```text
tests/
├── test_mediapipe.py
└── test_mediapipe_images.py
```

Run the tests from the project root:

```bash
pytest -v
```

The tests are intended to verify the MediaPipe detector and the image-based fall-detection workflow.

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
mediapipe==1.0.1
opencv-python
numpy
pytest
```

The MediaPipe model file must also be present:

```text
pose_landmarker_full.task
```

## MediaPipe Processing Pipeline

The processing performed by this branch can be represented as:

```text
Input Image
     │
     ▼
MediaPipe Pose Landmarker
     │
     ▼
Pose Landmarks
     │
     ├── Shoulders
     │
     └── Hips
     │
     ▼
Shoulder & Hip Midpoints
     │
     ▼
Torso Angle Calculation
     │
     ▼
Compare Against Threshold
     │
     ├── Angle < Threshold
     │       │
     │       ▼
     │      FALL
     │
     └── Angle ≥ Threshold
             │
             ▼
          NO_FALL
```

If no pose is detected, the system reports:

```text
NO_PERSON
```

## Role in the Overall AI System

The MediaPipe branch provides the **pose-based AI detection technique** for the project.

The camera/input component supplies an image frame, MediaPipe extracts the human pose landmarks, and the fall-detection component evaluates the torso orientation.

The intended integration flow is:

```text
Camera
   │
   ▼
OpenCV Frame
   │
   ▼
MediaPipe Pose Detection
   │
   ▼
Pose Landmarks
   │
   ▼
Torso Angle Calculation
   │
   ▼
Fall Classification
   │
   ▼
System Indicator / Response
```

This branch therefore focuses on the AI detection and classification component rather than the final integrated system.

## Limitations of the Current Implementation

- The current `MediaPipePoseDetector` default is configured for one pose.
- `get_landmarks()` uses only the first detected pose, even though the detector can be configured with a higher `num_poses` value. fileciteturn1file1L17-L23 fileciteturn1file1L96-L110
- Fall classification is based primarily on torso orientation.
- A person whose torso angle is above the selected threshold will not be classified as a fall.
- If the required shoulder or hip landmarks are unavailable or have insufficient visibility, a torso angle cannot be calculated.
- The parameter results are based on a 20-image test dataset and therefore do not establish performance across all possible real-world scenarios.

## Summary

The `feature/mediapipe` branch implements a pose-based fall-detection approach using the MediaPipe Pose Landmarker.

The branch:

1. Detects human pose landmarks.
2. Obtains the shoulder and hip landmarks.
3. Calculates the torso orientation.
4. Applies a torso-angle threshold.
5. Classifies the pose as `FALL`, `NO_FALL`, or `NO_PERSON`.
6. Tests multiple detection-confidence and torso-angle threshold combinations.
7. Produces raw, image-level, and parameter-combination CSV results for evaluation.

The branch is intended to provide the MediaPipe AI component that can later be connected to the camera input and system-level integration.
