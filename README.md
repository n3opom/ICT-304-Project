# ICT304 Intelligent Home Care System - YOLO Prototype

**Branch:** `feature/yolo`  
**Status:** Early experimental implementation; not the final live fall-detection application.

## Purpose

This branch explores a YOLO-based approach to identifying possible falls in still images. It uses a pretrained **YOLO11n object detector** (`yolo11n.pt`) to locate people, then applies a separate, hand-written rule to each person's bounding box. The rule compares the box's width-to-height ratio against a chosen threshold.

**Important:** YOLO identifies objects; it does not directly recognise fall events in this branch. The `FALL`/`NOT FALL` decision comes from the bounding-box ratio rule, not from a separately trained fall-classification model. This prototype does not implement temporal movement analysis, a continuous live-camera application, or caregiver alerts.

## Repository contents

```text
ICT-304-Project/
|-- data/
|   `-- test/                        # 20 labelled test images
|-- results/
|   |-- yolo_confidence_raw_results.csv
|   |-- yolo_image_results.csv
|   |-- yolo_combination_results.csv
|   |-- yolo_raw_results.csv
|   `-- yolo_threshold_results.csv
|-- src/
|   |-- yolo.py                      # YOLO wrapper with person filtering
|   |-- yolo_detection.py            # Alternative YOLO detection wrapper
|   `-- yolo_fall_detection.py       # Bounding-box ratio and threshold rule
|-- tests/
|   |-- test_yolo.py                 # Basic detector tests
|   `-- test_yolo_images.py          # Image-based parameter experiment
|-- ground_truth.csv                 # Image names and activity labels
|-- yolo11n.pt                       # Pretrained YOLO11n weights
|-- requirements.txt
`-- README.md
```

The repository also contains historical result files. The **current** `tests/test_yolo_images.py` script directly writes `yolo_confidence_raw_results.csv`, `yolo_image_results.csv`, and `yolo_combination_results.csv`. The other two result files are retained experiment artifacts, not outputs produced by that script.

## How detection works

1. `src/yolo.py` loads the pretrained YOLO11n model and runs object detection on an image.
2. `YOLODetector.detect_persons()` keeps only detections whose class name is `person`.
3. `calculate_bbox_ratio()` in `src/yolo_fall_detection.py` calculates `(x2 - x1) / (y2 - y1)` from each person's bounding box. A box with nonpositive height returns a ratio of `0.0`.
4. `classify_fall(ratio, threshold)` returns `True` when `ratio >= threshold`.
5. In the image experiment, an image is predicted as `FALL` if **any** detected person meets the ratio threshold. An image with no detected people is predicted `NOT FALL`.

A wide bounding box may be consistent with a horizontal posture, but it does not prove that a person has fallen. Camera viewpoint, occlusion, sitting, lying down, and bounding-box accuracy can affect this heuristic.

## Test images and ground truth

- Images are read from `data/test/`.
- `ground_truth.csv` maps image filenames to their labelled condition using the columns `image,condition`.
- This branch includes **20 images** and **20 ground-truth entries**: 10 labelled `falling` and 10 labelled with non-fall conditions (`sitting` or `standing`).
- The image experiment expects these exact counts. Replacing or extending the dataset requires updating the assertions in `tests/test_yolo_images.py`.

Only use images for which you have appropriate permission. Do not upload private participant images without their consent.

## Installation

Run commands from the repository root, where `requirements.txt` and `yolo11n.pt` are located. A compatible Python installation is required.

For Windows PowerShell, an optional virtual environment can be created **outside the repository**:

```powershell
py -m venv ..\ict304-yolo-venv
..\ict304-yolo-venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively, install into an existing Python environment:

```powershell
py -m pip install -r requirements.txt
```

The direct branch dependencies are:

- `ultralytics` - YOLO11n inference
- `opencv-python` - OpenCV image support
- `pytest` - basic tests and the image experiment

The pretrained `yolo11n.pt` model file is already included. You do not need to train a YOLO model to run these tests.

## Running tests and experiments

From the project root:

```powershell
python -m pytest tests/test_yolo.py -v
```

This runs three basic tests covering model loading, rejection of `None` input, and confidence configuration.

To run the complete image experiment:

```powershell
python -m pytest tests/test_yolo_images.py -v -s
```

The `-s` flag displays the printed experiment results. The experiment tests:

- YOLO confidence thresholds: `0.5`, `0.4`, `0.3`, `0.2`, `0.1`
- Fall ratio thresholds: `0.7`, `0.8`, `0.9`, `1.0`, `1.1`, `1.2`, `1.3`, `1.4`

This produces **40 parameter combinations** and **800 image/parameter classification cases** (20 images x 40 combinations). The program calculates true positives, true negatives, false positives, false negatives, accuracy, precision, recall, and F1-score for each combination.

To run both test files:

```powershell
python -m pytest tests/ -v -s
```

**Note:** The image experiment performs model inference across the test images, rather than being a small isolated unit test. It rewrites the three current experiment CSV outputs in `results/`.

## Results files

| File | Meaning |
| --- | --- |
| `results/yolo_confidence_raw_results.csv` | Person detections and bounding-box ratios at each confidence setting |
| `results/yolo_image_results.csv` | Prediction and TP/TN/FP/FN outcome for each image/parameter combination |
| `results/yolo_combination_results.csv` | Aggregated confusion counts and classification metrics for the 40 combinations |
| `results/yolo_raw_results.csv` | Historical image-level detection results retained in this branch |
| `results/yolo_threshold_results.csv` | Historical threshold-level evaluation results retained in this branch |

The current experiment writes the first three files above. Run the test again to regenerate those results using the code and dataset in your local checkout.

## Limitations and project relationship

- The width-to-height threshold is a manually selected heuristic, not a learned fall classifier.
- A single still image cannot establish whether a person underwent a rapid fall or deliberately lay down.
- Incorrect or missing person bounding boxes can lead to incorrect classifications.
- The 20-image dataset is small and does not establish performance in unseen home environments.
- This branch contains no `main.py` live application and does not send caregiver notifications.

This branch preserves an earlier YOLO investigation for comparison with the MediaPipe pose-based approach. The **official live Random Forest V3 implementation** is maintained separately in `main` and `feature/final-implementation`. This branch should not be mistaken for that completed implementation.
