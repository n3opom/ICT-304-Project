# ICT304 Intelligent Home Care System

## Main branch: Final Random Forest V3 fall-detection application

This branch contains the final **live-camera implementation** of the ICT304 AI System Design project. The application uses MediaPipe Pose to estimate human body landmarks, extracts posture and movement features, and uses a pretrained Random Forest V3 classifier to identify fall-like movement. Confirmed potential fall events are shown on screen and can trigger a Telegram caregiver notification.

This is an academic prototype, not a medical device or an emergency-response service. An alert indicates possible fall-like movement; it does not independently establish that a person has fallen or requires medical assistance.

The training scripts, offline evaluators and ground-truth annotations are kept separately in `feature/classifier-training-evaluation`. Earlier experiments and development history are retained on their respective feature branches.

## System workflow

1. Capture webcam frames through OpenCV (`src/camera.py`).
2. Estimate one person's 33 pose landmarks using the pretrained MediaPipe Pose Landmarker (`src/mediapipe.py`).
3. Apply three-frame median smoothing to the landmark coordinates when three frames are available (`LandmarkMedianFilter` in `main.py`). The initial frames are passed through without median smoothing.
4. Extract 11 posture-related measurements from nine selected landmarks, then form 66 temporal features from a rolling observation window (`src/fall_features.py`).
5. Use the pretrained Random Forest V3 model to estimate a fall probability (`src/ml_fall_detector.py`).
6. Confirm potential falls with temporal alert logic, display the current status, optionally send a Telegram notification and write local CSV diagnostics.

The detector uses the following settings in the supplied code:

| Setting | Value |
| --- | --- |
| Requested camera resolution | 1280 x 720 |
| Requested camera frame rate | 30 FPS |
| Fall probability threshold | 0.50 |
| Positive predictions required | 2 within 0.5 seconds |
| Detector alert cooldown | 8 seconds |
| On-screen alert duration | 3 seconds |
| Caregiver notification cooldown | 60 seconds |

The webcam's actual resolution and frame rate depend on device support. The classifier uses a one-second feature-history window and the sampling frequency stored in its model file.

## Repository contents

```text
ICT-304-Project/
|-- main.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- models/
|   |-- fall_classifier_v3.joblib
|   `-- pose_landmarker_full.task
`-- src/
    |-- __init__.py
    |-- alert_notifier.py
    |-- camera.py
    |-- fall_features.py
    |-- landmark_logger.py
    |-- mediapipe.py
    `-- ml_fall_detector.py
```

**Key components:**

- `main.py`: Application entry point, landmark smoothing, webcam display and subsystem orchestration.
- `src/camera.py`: Camera input handling.
- `src/mediapipe.py`: MediaPipe Pose Landmarker wrapper.
- `src/fall_features.py`: The 11 per-frame measurements and 66 temporal features used for live inference.
- `src/ml_fall_detector.py`: V3 model loading, fall probability calculation and temporal alert confirmation.
- `src/alert_notifier.py`: Telegram or simulated notification handling and notification event logging.
- `src/landmark_logger.py`: CSV logging of raw and median-filtered landmarks.
- `models/`: Required pretrained MediaPipe and V3 Random Forest model assets.

## Prerequisites

- A computer with a functioning webcam (the project was developed using Windows).
- Python 3 and `pip`; Python 3.13 is the version used during the project's Windows development.
- The two supplied files inside `models/`.
- Internet access and a configured Telegram bot/chat if using the default Telegram notification mode.

Run the commands below from the repository root (the folder containing `main.py`).

## Installation

### Windows PowerShell

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, either adjust your local execution policy appropriately or use `venv\Scripts\python.exe` in place of `python` for installation and execution.

### Git Bash on Windows

```bash
py -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `venv/` directory is ignored by the supplied `.gitignore`.

## Configure notifications

**Default behaviour:** `main.py` creates `AlertNotifier(mode="telegram", cooldown_seconds=60.0)`. The application requires both environment variables below to be set before it can start:

- `TELEGRAM_BOT_TOKEN`: Token for a Telegram bot you control.
- `TELEGRAM_CHAT_ID`: Destination chat ID authorised to receive notifications.

For **PowerShell**, set them in your current terminal session:

```powershell
$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
$env:TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
```

For **Git Bash**, use:

```bash
export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
export TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
```

Do **not** commit real bot tokens or chat IDs to GitHub.

For testing without Telegram, change the notifier construction in `main.py` from `mode="telegram"` to `mode="simulated"` locally. The simulated mode records and prints a notification instead of contacting Telegram. This is an optional local configuration change, not the default state of this branch.

## Run the application

With the environment activated and notification mode configured:

```bash
python main.py
```

Or, from a Windows terminal using the Python launcher:

```powershell
py main.py
```

The application requests webcam index `0`. If the wrong camera opens or no webcam is detected, review `CAMERA_INDEX` near the top of `main.py`. Press **Q** while the video window is active to exit normally.

The window displays detected pose landmarks, the torso-angle measurement when available, the fall probability and one of these statuses:

- `NO PERSON DETECTED`
- `NO FALL ALERT`
- `FALL-LIKE MOVEMENT`
- `POTENTIAL FALL ALERT`

A single positive classification does not automatically trigger a caregiver notification; the alert-confirmation and cooldown conditions must be satisfied.

## Local output files and privacy

The application creates the `results/` directory as needed:

| Output | Description |
| --- | --- |
| `results/landmarks.csv` | Landmark data for detected poses, including raw and filtered coordinates, visibility, timestamp and current display state. Recreated when the logger starts. |
| `results/caregiver_alerts.csv` | Appended notification attempts and statuses for potential-fall alerts. |

These CSV files are local runtime output and are excluded from Git by the supplied `.gitignore`. They may nevertheless contain sensitive information about recorded movements. Handle and share them only with appropriate consent. The default Telegram mode sends text notifications to the configured chat; the source code does not attach webcam video to those messages.

The original training and participant recordings are not part of this branch. Training and offline evaluation resources are organised in `feature/classifier-training-evaluation`, with original participant videos withheld from public GitHub publication for privacy.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`:** Open a terminal in the repository root and run `python main.py` from there.
- **`Missing Telegram environment variables`:** Set both Telegram variables in the same terminal session, or use simulated mode for local testing.
- **`Could not open webcam` / `Failed to read frame`:** Check camera permissions, close other applications using the camera and verify `CAMERA_INDEX`.
- **Missing `.task` or `.joblib` model:** Verify both required model files exist in `models/` with the filenames listed above.
- **Dependency or model deserialization issue:** Reinstall the branch's requirements in a fresh virtual environment and use only trusted model files. The `joblib` model is a serialized Python object and should not be loaded from untrusted sources.

## Scope and limitations

This branch is designed for **live inference**, not for retraining or cross-validation. Classification and alert results depend on pose visibility, camera placement, movement patterns, and the data represented during model development. The system can generate false alerts or miss real falls and should not be relied upon as a sole safety measure.

For the training, evaluation and dataset preparation procedures, refer to the `feature/classifier-training-evaluation` branch.
