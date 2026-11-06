# ICT304 Intelligent Home Care System — Temporal Fall Detection

## Branch purpose

This is the `feature/temporal-fall-detection` development branch of the ICT304 AI System Design project. Its **live application uses a MediaPipe Pose-based, rule-driven temporal fall detector**, not the final deployed Random Forest V3 classifier.

This branch explores how posture, movement speed, recent pose history, and sustained lying can be combined to distinguish possible falls from controlled movements. It also retains experimental Random Forest training and evaluation scripts from the broader development process. Those scripts are **separate from the live temporal detector in `main.py`**.

For the final Random Forest V3 live system, see the repository's `main` or `feature/final-implementation` branch. For the isolated training and evaluation workflow, see `feature/classifier-training-evaluation`.

## Live processing pipeline

```text
Laptop webcam
    -> src/camera.py (OpenCV frames)
    -> src/mediapipe.py (MediaPipe Pose Landmarker)
    -> src/temporal_fall_detector.py
         -> three-frame median coordinate smoothing
         -> torso/body-position measurements and posture estimation
         -> temporal fall-state rules
    -> on-screen posture, torso angle and fall state
    -> src/landmark_logger.py (results/landmarks.csv)
```

The live code takes the first detected person's 33 MediaPipe pose landmarks. It displays one of three fall states: `NO FALL`, `POSSIBLE FALL`, or `FALL`. It also estimates postures such as standing, sitting, crouching, lying, or uncertain. This branch does **not** send Telegram, email or emergency-service notifications; its output is the OpenCV display and local CSV landmark log.

### Temporal behaviour

The rule-based detector:

- Applies a rolling three-frame median to landmark coordinates after the filter has received three frames. Earlier frames pass through without median substitution.
- Uses landmark visibility, torso orientation, the person's posture, and time-stamped observations.
- Can raise `POSSIBLE FALL` when a rapid upright-to-lying transition is observed. Its code checks for an upright observation within one second and an angle-change rate of at least 90 degrees per second.
- Can confirm `FALL` when a transition-based candidate is followed by sustained lying for the configured confirmation period (one second by default).
- Can raise `POSSIBLE FALL` after prolonged unexplained lying without automatically promoting that case to `FALL` on duration alone.
- Handles missing/uncertain pose data, candidate cancellation, and recovery to an upright posture.

The `POSSIBLE_FALL_ANGLE` and `RAPID_MOVEMENT_THRESHOLD` settings in `main.py` are passed into the detector, but the current temporal decision logic does not independently use those settings as fall triggers. This branch is a research prototype; its heuristic thresholds are not clinically validated.

## Repository structure

```text
ICT-304-Project/
├── main.py                         # Live rule-based temporal demonstration
├── src/
│   ├── camera.py                   # Webcam/image input component
│   ├── mediapipe.py                # MediaPipe Pose Landmarker wrapper
│   ├── temporal_fall_detector.py   # Median smoothing, posture and temporal rules
│   ├── landmark_logger.py          # Raw/filtered landmark CSV logging
│   └── mediapipe_fall_detection.py # Earlier posture-based investigation
├── tests/
│   ├── test_camera.py
│   ├── test_mediapipe.py
│   └── test_temporal_fall_detector.py
├── models/
│   ├── pose_landmarker_full.task   # Required by live application
│   ├── fall_classifier.joblib      # Historical Random Forest V1 model
│   ├── fall_classifier_v2.joblib   # Historical Random Forest V2 model
│   └── fall_classifier_v3.joblib   # Historical Random Forest V3 model
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

`training_videos/` and `results/` may be absent after downloading the repository. The scripts use or create these local folders as needed. No video recordings are supplied in the public repository.

## Setup

Run commands from the repository root, where `main.py` and `requirements.txt` are located.

1. Install a supported Python version and make sure `py` or `python` is available from your terminal. Use a Python environment compatible with the pinned MediaPipe release.
2. Create and activate a virtual environment (Windows PowerShell):

   ```powershell
   py -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install the packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Confirm that `models/pose_landmarker_full.task` exists.
5. Connect a webcam and allow camera access in Windows.

### Run the live detector

```powershell
python main.py
```

The default camera index is `0`. The program requests 1280 × 720 resolution and 30 FPS, although the actual capture settings depend on the webcam and driver. Press **Q** in the OpenCV window to quit.

If the camera cannot be opened, close other software using it and check `CAMERA_INDEX` in `main.py`. To adjust temporal rules, inspect the constants in `main.py` and the implementation in `src/temporal_fall_detector.py` before changing them.

### Live output

The application draws the filtered pose landmarks and displays its estimated posture, fall state and torso angle. It writes raw and filtered landmark coordinates, visibility/presence values, posture and fall state to `results/landmarks.csv`.

This CSV is opened in write mode on each run, so an existing file with the same name will be overwritten. Copy it elsewhere first if you need to preserve a previous session. Generated `results/*.csv` files are ignored by Git.

## Automated tests

This branch contains pytest tests for the camera, MediaPipe wrapper and temporal fall detector. Run:

```powershell
python -m pytest -q tests
```

Tests cover examples including posture classification, rapid and gradual transitions, missing landmarks, reset behaviour and median-filter rejection of coordinate spikes. They do not by themselves establish real-world fall-detection accuracy. Camera- and MediaPipe-related tests may depend on local hardware, model availability and the installed environment.

## Additional training and evaluation scripts

The branch also retains **separate** Random Forest V1, V2 and V3 experimentation scripts. They are not used by the live temporal `main.py`:

| Script | Purpose |
| --- | --- |
| `train_fall_classifier.py` | Random Forest V1 training |
| `train_fall_classifier_v2.py` | Random Forest V2 training |
| `train_fall_classifier_v3.py` | Random Forest V3 cross-validation and final-model training |
| `evaluate_fall_alerts.py` | V1 offline event-level evaluation |
| `evaluate_fall_alerts_v2.py` | V2 offline event-level evaluation |
| `evaluate_fall_alerts_v3.py` | V3 offline event-level evaluation |
| `evaluate_rule_based.py` | Recorded-video evaluation of the rule-based temporal detector |

These experiments use separate data and evaluation procedures. Do not treat their Random Forest metrics as results produced by the live temporal detector.

### Dataset privacy and local reproduction

The original participant videos are **not published on GitHub** to protect the test subjects' privacy. The project recordings will be submitted directly to Murdoch University's Learning Management System (LMS) for assessment.

The repository includes `ICT304_Project_GroundTruthTable.xlsx`, which documents the original labelled intervals. The source recordings required to reproduce those exact experiments are withheld. If you wish to run the scripts with your own data:

1. Create a folder named `training_videos/` in the repository root.
2. Record your own videos with appropriate permission from the participants and place the files in that folder. Do not publish identifiable recordings without permission.
3. Prepare and label your own `.xlsx` ground-truth workbook matching your videos. The training scripts look for `ICT304_Project_GroundTruthTable.xlsx` and require the columns `video`, `start_sec`, `end_sec`, and `label`. Filenames must match the MP4 files exactly; time intervals must match the recordings. Use labels supported by the scripts, such as `FALL_TRANSITION`, `POST_FALL`, `CONTROLLED_LYING`, `LYING_NO_FALL`, `WALKING`, and `RECOVERY`.
4. Update any fixed evaluation-video filenames or other dataset-specific settings in the experiment scripts when using a new dataset. In particular, the V1/V2 replay evaluators contain hard-coded test-video names.

For example, after preparing a compatible local dataset, the training commands are:

```powershell
python train_fall_classifier.py
python train_fall_classifier_v2.py
python train_fall_classifier_v3.py
```

Evaluation commands are:

```powershell
python evaluate_rule_based.py
python evaluate_fall_alerts.py
python evaluate_fall_alerts_v2.py
python evaluate_fall_alerts_v3.py
```

**Important:** The training/evaluation scripts cannot be fully reproduced using the public repository alone because the original labelled video files are not included. The existing workbook is not a substitute for the corresponding source videos.

## Limitations

- This branch's live application uses manually defined temporal rules rather than a trained fall classifier.
- Detection relies on visible body landmarks and a suitable camera viewpoint; lighting, occlusion, tracking loss and deliberate lying movements may affect its output.
- Fall states are visual indicators, not an automatically dispatched caregiver alert.
- The historical Random Forest scripts and models are retained for development context but are not part of this branch's live inference path.
- The software is an experimental academic prototype and is not a validated emergency response system.
