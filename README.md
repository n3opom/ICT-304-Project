# ICT304 Intelligent Home Care System — Final Implementation

This branch contains the **standalone, webcam-based Random Forest V3 fall-detection application** for the ICT304 AI System Design project. It is the live inference implementation, not the model-training or offline-evaluation workspace.

The application detects a person's body pose, analyses recent posture and movement, displays a fall classification score, and can send a **potential fall** notification through Telegram. It is an academic prototype, not a medical device or emergency-response service.

## What this branch includes

- Live webcam capture through OpenCV (camera index `0` by default).
- MediaPipe Pose Landmarker inference with the bundled `pose_landmarker_full.task` model.
- **Three-frame median smoothing** of landmark coordinates before feature extraction. The first two detected frames are passed through without median smoothing while the history fills; the filter resets when no pose is detected.
- Nine selected pose landmarks used to calculate 11 measurements per processed observation.
- A one-second temporal feature window summarised into **66 features** for the pretrained Random Forest V3 classifier.
- On-screen fall probability, torso angle, pose skeleton and monitoring status.
- Potential-fall alerts with temporal confirmation and cooldown periods.
- Telegram caregiver alerts (or a local simulated notification mode).
- CSV logging of raw and filtered pose landmarks, and notification outcomes.

This branch **does not contain** the V1–V3 training scripts, offline evaluation scripts, private training videos or ground-truth workbook. Those experiments are maintained separately in `feature/classifier-training-evaluation`.

## Directory structure

```text
ICT-304-Project/
├── main.py
├── models/
│   ├── fall_classifier_v3.joblib
│   └── pose_landmarker_full.task
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── mediapipe.py
│   ├── fall_features.py
│   ├── ml_fall_detector.py
│   ├── alert_notifier.py
│   └── landmark_logger.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `results/` directory and its CSV outputs are created locally at runtime and are not committed to this branch.

## Requirements

- Python and `pip` (the application was developed for Windows using Python 3.13).
- A working webcam and a desktop environment capable of displaying an OpenCV window.
- Network access and valid Telegram bot credentials **when using the default Telegram notification mode**.
- The two model files shown above, already included in this branch.

Install Python packages from `requirements.txt`. Training-related packages such as `pandas`, `openpyxl` and `pytest` are not needed to launch this live application.

## Installation and setup

From the repository directory, select this branch (if it has not already been checked out):

```bash
git switch feature/final-implementation
```

Create and activate a virtual environment in PowerShell:

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, use Git Bash:

```bash
py -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, use `venv\Scripts\python.exe -m pip install -r requirements.txt` and `venv\Scripts\python.exe main.py` directly instead.

### Telegram notification setup

The current `main.py` creates `AlertNotifier(mode="telegram", cooldown_seconds=60.0)`. **Without both environment variables below, startup will fail.**

Set them in the **same terminal session** before launching the program. In PowerShell:

```powershell
$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
$env:TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
```

Or in Git Bash:

```bash
export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
export TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
```

Obtain your bot token from a bot you control, and configure the destination chat ID for that bot. Never commit actual tokens, chat IDs or other secrets to GitHub.

If Telegram is unavailable and only a local demonstration is needed, the notifier supports `mode="simulated"`. To use it, change **only the notifier's `mode` argument** in your local `main.py` from `"telegram"` to `"simulated"`. This is an optional local configuration change; the committed branch defaults to Telegram.

## Run the live application

From the project root, with the environment activated and (for Telegram mode) credentials configured:

```bash
python main.py
```

On Windows, `py main.py` also works when the appropriate packages are installed for the Python interpreter selected by `py`.

The application requests a camera resolution of **1280 × 720 at 30 FPS**, although the device may supply a different actual rate or resolution. Change `CAMERA_INDEX` near the top of `main.py` if the intended webcam is not at index `0`.

Press **Q** while the preview window is active to stop monitoring. On shutdown, camera and MediaPipe resources are released and the notifier attempts to send a monitoring-stopped status message in Telegram mode.

## Processing flow

```text
Webcam frame
    -> MediaPipe Pose Landmarker (up to one person, 33 landmarks)
    -> Three-frame landmark-coordinate median filter
    -> Nine selected landmarks and 11 per-frame measurements
    -> One-second history / 66 temporal features
    -> Random Forest V3 fall probability
    -> Temporal confirmation and alert state
    -> Screen display, caregiver notification, CSV logging
```

The preview draws the **raw** pose landmarks. Classification and the displayed torso-angle measurement use the **filtered** landmarks. The landmark CSV retains both raw and filtered coordinates.

### Detection and alert settings

| Setting | Value | Location |
| --- | --- | --- |
| Fall probability threshold | 0.50 | `src/ml_fall_detector.py` |
| Minimum positive predictions | 2 within the evidence window | `src/ml_fall_detector.py` |
| Positive evidence window | 0.5 seconds | `src/ml_fall_detector.py` |
| Detector alert cooldown | 8 seconds | `src/ml_fall_detector.py` |
| On-screen alert display | 3 seconds | `src/ml_fall_detector.py` |
| Caregiver notification cooldown | 60 seconds | `main.py` / `src/alert_notifier.py` |

The two positive predictions need not be consecutive. A fall probability above the threshold is a **fall-like prediction**, not an independently verified emergency. The program displays `FALL-LIKE MOVEMENT` while a positive prediction is present and `POTENTIAL FALL ALERT` when the alert state is active.

## Runtime output

| File | Contents |
| --- | --- |
| `results/landmarks.csv` | One row per detected landmark, including timestamp, frame number, raw and filtered coordinates, visibility, presence, display status and torso angle. This file is **overwritten** at the start of each run. |
| `results/caregiver_alerts.csv` | Notification attempts and their outcomes (`SENT`, `FAILED` or `SIMULATED`, depending on mode). This file is **appended** across runs. |

The `.gitignore` excludes generated CSV files under `results/`. Depending on the data recorded, landmark logs may reveal sensitive information about monitored people; handle and share them appropriately.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**: Run the command from the repository's root directory (the directory containing `main.py`).
- **Missing Telegram environment variables**: Configure both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the terminal before running, or switch locally to simulated mode.
- **Camera cannot be opened**: Check Windows camera permissions, close other webcam applications and verify `CAMERA_INDEX`.
- **Missing model file**: Confirm that both model files are present in `models/` under their exact names.
- **Fall probability unavailable**: The temporal feature extractor needs at least five timestamped observations with a positive time interval, and a detectable pose must be present.
- **Telegram notification failed**: Confirm internet connectivity, bot token, chat ID and that the bot can message the destination; review the console and `results/caregiver_alerts.csv`.

## Scope and limitations

This system monitors **one detected person at a time** and relies on visible pose landmarks. Occlusion, lighting, camera angle, controlled lying-down movements and differences between individuals can affect predictions. False positives and missed falls remain possible. The Telegram message is a **prototype caregiver notification**, not a call to emergency services. The application should not be relied on as the sole means of monitoring a person at risk.

For the model-training procedures, cross-validation experiments, ground-truth annotations and offline evaluation results, refer to `feature/classifier-training-evaluation`. Its original participant recordings are intentionally not distributed publicly for privacy reasons.
