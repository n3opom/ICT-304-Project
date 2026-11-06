# ICT304 Intelligent Home Care System — Random Forest Integration

**Branch:** `feature/random-forest-integration`  
**Purpose:** Preserve the integrated fall-detection development code and experimental implementations.

## Overview

This branch contains the integrated camera-based fall-detection project developed for ICT304 AI System Design. It preserves the working Random Forest V3 live application alongside earlier classifiers, training and replay evaluators, and legacy rule-based fall-detection code.

**This is a development and integration archive, not the recommended deployment branch.** The official live application is maintained on `main` and `feature/final-implementation`; the separated machine-learning experiment code is maintained on `feature/classifier-training-evaluation`. The YOLO investigation is maintained on `feature/yolo` and is not part of the V3 live pipeline in this branch.

## What this branch contains

| Component | Files | Role |
| --- | --- | --- |
| Live application | `main.py` | Webcam capture, pose display, V3 prediction, alerts and logging |
| Camera and pose estimation | `src/camera.py`, `src/mediapipe.py` | OpenCV video input and MediaPipe pose landmarks |
| Live preprocessing | `src/fall_features.py`, `src/ml_fall_detector.py` | Temporal features, Random Forest V3 inference and alert confirmation |
| Notifications and logs | `src/alert_notifier.py`, `src/landmark_logger.py` | Telegram or simulated notifications and CSV outputs |
| Training | `train_fall_classifier.py`, `train_fall_classifier_v2.py`, `train_fall_classifier_v3.py` | Random Forest V1, V2 and V3 training and classification evaluation |
| Replay evaluation | `evaluate_fall_alerts.py`, `evaluate_fall_alerts_v2.py`, `evaluate_fall_alerts_v3.py` | Recorded-video alert-event evaluation |
| Historical rule-based code | `src/mediapipe_fall_detection.py`, `src/temporal_fall_detector.py`, `evaluate_rule_based.py` | Earlier development and comparison only; not the live V3 classifier |
| Automated tests | `tests/` | Camera, MediaPipe and earlier temporal rule-based detector tests |
| Model assets | `models/` | MediaPipe pose model and saved V1–V3 Random Forest models |
| Label reference | `ICT304_Project_GroundTruthTable.xlsx` | Annotations associated with the original, privately held recordings |

The final live application **does not import the training scripts**. Its feature extraction is provided by `src/fall_features.py`.

## Live V3 processing pipeline

```text
Webcam frame
    -> MediaPipe Pose (up to 33 body landmarks)
    -> Three-frame median smoothing of landmark coordinates
    -> Nine selected landmarks and 11 per-frame measurements
    -> Temporal window and 66 numerical features
    -> Random Forest V3 fall probability
    -> Temporal alert confirmation
    -> On-screen status, notification and CSV logs
```

`main.py` requests a 1280 x 720 webcam feed at 30 FPS with camera index `0`; actual camera performance depends on the device. The three-frame filter uses unfiltered coordinates until three frames are available and resets when no landmarks are detected. The on-screen skeleton uses the detected pose landmarks, while the classifier uses the filtered landmarks.

The V3 detector uses a fall-probability threshold of **0.50**, requires **two positive predictions within 0.5 seconds**, applies an **8-second detector alert cooldown**, and displays an active alert for **3 seconds**. Separately, the notifier in `main.py` has a **60-second notification cooldown**.

A fall-like prediction is a potential alert, not a medical diagnosis or confirmation of an emergency.

## Repository layout

```text
ICT-304-Project/
├── main.py
├── src/
│   ├── camera.py
│   ├── mediapipe.py
│   ├── fall_features.py
│   ├── ml_fall_detector.py
│   ├── landmark_logger.py
│   ├── alert_notifier.py
│   ├── mediapipe_fall_detection.py
│   └── temporal_fall_detector.py
├── models/
│   ├── pose_landmarker_full.task
│   ├── fall_classifier.joblib
│   ├── fall_classifier_v2.joblib
│   └── fall_classifier_v3.joblib
├── tests/
├── train_fall_classifier.py
├── train_fall_classifier_v2.py
├── train_fall_classifier_v3.py
├── evaluate_fall_alerts.py
├── evaluate_fall_alerts_v2.py
├── evaluate_fall_alerts_v3.py
├── evaluate_rule_based.py
├── ICT304_Project_GroundTruthTable.xlsx
├── requirements.txt
└── README.md
```

`training_videos/` is intentionally absent from the published branch. `results/` is generated locally by the application and experiment scripts, and result CSVs are ignored by the provided `.gitignore` unless explicitly added.

## Installation

Use a Python environment compatible with the versions of MediaPipe and other packages listed in `requirements.txt`. From the repository root:

### Windows PowerShell

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Git Bash on Windows

```bash
py -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The four model files shown above must remain in `models/`. A connected webcam is needed for live operation.

## Run the integrated live application

By default, `main.py` creates `AlertNotifier(mode="telegram")`. Supply the credentials as environment variables before starting it; **do not put bot credentials in the repository**.

PowerShell:

```powershell
$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
$env:TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
python main.py
```

Git Bash:

```bash
export TELEGRAM_BOT_TOKEN='YOUR_BOT_TOKEN'
export TELEGRAM_CHAT_ID='YOUR_CHAT_ID'
python main.py
```

Press **Q** in the video window to exit. A normal shutdown also sends a monitoring-stopped notification in Telegram mode.

For local operation without Telegram, the notifier supports `mode="simulated"`; this requires changing its configuration in `main.py` on your local copy. Without valid Telegram variables, the default configuration raises an error at startup.

Runtime output files:

- `results/landmarks.csv` — raw and median-filtered landmark coordinates and frame diagnostics; recreated for each live run.
- `results/caregiver_alerts.csv` — appended notification attempts and statuses.

These CSVs may contain sensitive observation data; review them before sharing.

## Training and evaluating Random Forest models

**Training requires local MP4 recordings and a matching Excel annotation workbook.** These are not interchangeable with live webcam input. The training scripts are run from the repository root:

| Task | Command | Details |
| --- | --- | --- |
| Train V1 | `python train_fall_classifier.py` | 66 temporal features; video-level holdout evaluation |
| Train V2 | `python train_fall_classifier_v2.py` | 102 temporal features; video-level holdout evaluation |
| Train V3 | `python train_fall_classifier_v3.py` | 66 features; five-fold video-level cross-validation, then full-data model training |
| Replay V1 alerts | `python evaluate_fall_alerts.py` | Uses predefined V1 held-out video names |
| Replay V2 alerts | `python evaluate_fall_alerts_v2.py` | Uses predefined V2 held-out video names |
| Replay V3 alerts | `python evaluate_fall_alerts_v3.py` | Reconstructs video-level folds and evaluates held-out alert events |

The V1 and V2 replay scripts contain experiment-specific `TEST_VIDEOS` lists; those filenames must match the available local dataset and an appropriate held-out split. V3 five-fold cross-validation requires sufficient videos with and without labelled fall transitions to form all five folds.

V3 training writes `models/fall_classifier_v3.joblib`, `results/v3_cross_validation.csv` and `results/v3_cv_predictions.csv`. Replay evaluators write `results/ml_replay_predictions.csv`, `results/ml_potential_alerts.csv` and `results/ml_false_alerts.csv`. **Replay scripts use overlapping output filenames**, so save or rename results before running a different version if you need to preserve each experiment separately.

The `evaluate_rule_based.py` script and `src/temporal_fall_detector.py` are historical investigations, **not** part of the live Random Forest V3 path.

## Participant privacy and local dataset setup

**Original participant videos are deliberately withheld from GitHub to protect test-subject privacy.** They will be submitted directly to **Murdoch University's Learning Management System (LMS)** for the academic assessment, rather than distributed through this public repository.

Anyone using this branch to train or independently evaluate a model must prepare their **own recordings and labels**:

1. Create a `training_videos/` directory in the repository root on your own device.
2. Record your own appropriately consented test footage and place the `.mp4` files in that directory. Do not upload private participant recordings to GitHub.
3. Prepare your own Excel ground-truth workbook named `ICT304_Project_GroundTruthTable.xlsx`, using the required columns `video`, `start_sec`, `end_sec` and `label`. The `video` entries must exactly match the local MP4 filenames. Start and end times are in seconds, must be valid and must not overlap for the same video.
4. Use activity labels accepted by the scripts, including `FALL_TRANSITION`, `POST_FALL`, `NORMAL`, `STANDING`, `WALKING`, `SITTING`, `CROUCHING`, `CONTROLLED_LYING`, `LYING_NO_FALL`, `RECOVERY`, `NON_FALL` and `BENDING`.
5. Keep a safe copy of the original workbook before replacing it with annotations for your own recordings. The workbook included here refers to the original private dataset and cannot be used to rerun the experiments without matching video files.

Because the original recordings are withheld, external users **cannot fully reproduce the original training and video-replay experiments from the GitHub branch alone**. The saved models can still be used for live inference without those private training videos.

## Automated testing

Run from the repository root:

```bash
python -m pytest tests -v
```

The included tests cover camera behaviour, MediaPipe usage and the earlier temporal rule-based detector. **They do not constitute comprehensive unit-test coverage of the deployed V3 Random Forest pipeline, alert notifier or CSV logger.** Recorded-video classification and alert replay are separate evaluation procedures, not replacements for targeted unit tests.

## Limitations and branch guidance

The system is an academic prototype. Its fall predictions can be affected by body occlusion, camera position, lighting and normal movements that resemble falls. Evaluation on the original recordings does not guarantee performance on new people or environments. The alert system is not a substitute for emergency services or professional supervision.

- **`main` / `feature/final-implementation`**: recommended live V3 implementation.
- **`feature/classifier-training-evaluation`**: separated training and evaluation code and experiment artefacts.
- **`feature/temporal-fall-detection`**: earlier temporal rule-based development.
- **`feature/yolo` / `feature/mediapipe`**: earlier technique investigations.
- **`feature/random-forest-integration` (this branch)**: broader development archive combining experimental and integrated code.

