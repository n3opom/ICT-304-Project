# ICT304 Intelligent Home Care System — MediaPipe Prototype

**Branch:** `feature/mediapipe`  
**Purpose:** Early pose-based fall-classification prototype and image-based parameter evaluation.

## Overview

This branch investigates whether body landmarks obtained from **MediaPipe Pose Landmarker** can support a simple fall/no-fall classification. It detects pose landmarks in an image, calculates the orientation of the torso using the shoulders and hips, and compares the angle against a configurable threshold.

This is an **experimental, image-based prototype**, not the project's final live monitoring application. The final deployed Random Forest V3 application is maintained separately in `main` and `feature/final-implementation`.

## Repository structure

```text
.
├── src/
│   ├── mediapipe.py                 # MediaPipe pose detection
│   └── mediapipe_fall_detection.py  # Torso-angle fall classification
├── tests/
│   ├── test_mediapipe.py            # Component checks
│   └── test_mediapipe_images.py     # Image-based parameter experiment
├── data/
│   └── test_images/                 # 20 labelled test images
├── results/
│   ├── mediapipe_raw_results.csv
│   ├── mediapipe_image_results.csv
│   └── mediapipe_combination_results.csv
├── ground_truth.csv                # Image names and activity labels
├── pose_landmarker_full.task       # Pretrained pose-landmarker model
├── requirements.txt
└── README.md
```

## How it works

1. `MediaPipePoseDetector` (`src/mediapipe.py`) converts an OpenCV BGR image to RGB and runs the pretrained MediaPipe Pose Landmarker in **IMAGE** mode.
2. The detector returns the landmark list for the **first detected pose**, or `None` if no pose is detected. The default configuration requests one pose.
3. `MediaPipeFallDetector` (`src/mediapipe_fall_detection.py`) finds the left/right shoulders and hips, checks their visibility, and calculates the shoulder and hip midpoints.
4. The torso angle relative to the horizontal is calculated from the midpoint-to-midpoint vector. An angle near **0°** is approximately horizontal; near **90°** is approximately vertical.
5. With the default `torso_angle_threshold=60.0`, a valid torso angle **below 60°** is classified as `FALL`; otherwise it is `NO_FALL`. The standalone classifier returns `NO_PERSON` if no landmarks are supplied.

The default minimum required landmark visibility is `0.5`. If required landmarks have insufficient visibility, the classifier cannot establish a falling posture and returns `NO_FALL`. The experiment also treats missing pose detections as negative (`NO_FALL`) predictions when calculating its confusion matrix.

**Important:** This rule describes torso orientation in a still image. It does **not** establish whether a person actually experienced a fall, and does not measure the speed or duration of the movement. Deliberate lying down may appear similar to a fall.

## Installation

From the repository root, create and activate a virtual environment (recommended):

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On Git Bash:

```bash
source venv/Scripts/activate
```

Install the branch dependencies:

```bash
python -m pip install -r requirements.txt
```

The supplied `pose_landmarker_full.task` file must remain in the **project root**, alongside `README.md`. This is the default path used by `src/mediapipe.py`.

## Run the tests and image experiment

Run the component checks:

```bash
python -m pytest tests/test_mediapipe.py -v
```

Run the image experiment:

```bash
python -m pytest tests/test_mediapipe_images.py -s
```

Or run all tests and the experiment together:

```bash
python -m pytest -v
```

The image experiment is also defined as `run_mediapipe_experiment()` inside `tests/test_mediapipe_images.py`. **No `main.py` or webcam application is provided on this branch.**

### Required test data

The experiment expects **exactly 20 images** inside `data/test_images/` and matching rows in `ground_truth.csv`. The included dataset has 10 images labelled `falling` and 10 labelled `sitting` or `standing`.

The ground-truth CSV uses these columns:

```csv
image,condition
fall001.jpg,falling
not fallen001.jpg,sitting
```

The `image` values must exactly match the filenames in `data/test_images/` (including spaces). Labels such as `falling` or `fall` are mapped to `FALL`; `sitting`, `standing`, `not fallen` and other supported non-fall labels map to `NO_FALL`.

If you substitute a different image dataset, you must update the ground-truth CSV accordingly. The current experiment also asserts that there are exactly 20 images, so a larger or smaller dataset requires modifying that test.

## Evaluation procedure

`tests/test_mediapipe_images.py` evaluates:

- **Five pose-detection confidence settings:** `0.3`, `0.4`, `0.5`, `0.6`, `0.7`.
- **Eight torso-angle thresholds:** `30°`, `35°`, `40°`, `45°`, `50°`, `55°`, `60°`, `65°`.
- **40 parameter combinations** across the 20 labelled images, producing 800 image-level classification rows.

Pose estimation is performed once per image for each confidence setting and cached before applying the different torso-angle thresholds. The experiment calculates true positives, true negatives, false positives, false negatives, accuracy, precision, recall and F1-score.

| Output | Contents | Expected rows |
|---|---|---:|
| `results/mediapipe_raw_results.csv` | Per-image pose detection and torso measurements at each confidence setting | 100 |
| `results/mediapipe_image_results.csv` | Individual classifications for every confidence/threshold combination | 800 |
| `results/mediapipe_combination_results.csv` | Confusion counts and aggregate metrics for each parameter combination | 40 |

These CSVs are **overwritten** when the experiment is rerun.

### Recorded results

The committed `mediapipe_combination_results.csv` includes configurations with **90% accuracy, 100% precision, 80% recall and an F1-score of approximately 88.89%** (TP = 8, TN = 10, FP = 0, FN = 2). These figures describe only this small 20-image test set and should not be interpreted as real-world fall-detection accuracy.

## Limitations and privacy

- This branch tests individual images, not continuous video or event-level fall detection.
- The classifier depends on visible shoulder and hip landmarks and uses only torso orientation for fall classification.
- The selected thresholds and results are based on a small, fixed collection of images; results may change with different people, camera angles and surroundings.
- The repository contains test images. Anyone adding or replacing images should ensure appropriate permission to use or publish images of identifiable participants. Avoid committing private recordings or other personal data.
- The final system's live detection, temporal processing, trained Random Forest classifier and caregiver notifications are **not** implemented by this branch.

## Relationship to the main project

This prototype was used to investigate MediaPipe pose landmarks as an alternative to bounding-box-based object detection. Subsequent project branches developed temporal analysis and a trained Random Forest V3 classifier. Refer to `main` for the final live implementation and to `feature/classifier-training-evaluation` for the later classifier training/evaluation code.
