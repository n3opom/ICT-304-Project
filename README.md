# ICT-304 Project — Integration Branch

## Overview

The `feature/integration` branch combines the project's camera input and MediaPipe Pose components into a single working prototype.

The branch is responsible for the **system-level integration** of:

- Live webcam input
- MediaPipe Pose detection
- Pose landmark visualisation
- Landmark data logging
- Basic automated testing of the integrated components

The current integration prototype focuses on obtaining a live webcam frame, passing it to MediaPipe Pose, displaying the detected pose landmarks, and recording the landmark data to a CSV file.

It does **not currently implement the final fall-classification logic** from the standalone MediaPipe fall-detection branch.

## Branch Structure

```text
ICT-304-Project/
│
├── .vscode/
│
├── models/
│   └── pose_landmarker_full.task
│
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── landmark_logger.py
│   └── mediapipe.py
│
├── tests/
│   ├── __init__.py
│   ├── test_camera.py
│   └── test_mediapipe.py
│
├── .gitignore
├── README.md
├── main.py
└── requirements.txt
```

## Integrated System Components

The integration branch combines three main components.

### 1. Camera

Implemented in:

```text
src/camera.py
```

The `Camera` class supports:

- `live` mode for webcam input
- `image` mode for pre-taken images

The integration prototype uses:

```python
Camera(
    mode="live",
    camera_index=0
)
```

The camera is opened and configured before frames are processed. fileciteturn5file7L5-L22 fileciteturn5file1L100-L128

### 2. MediaPipe Pose

Implemented in:

```text
src/mediapipe.py
```

The integration uses the MediaPipe Pose Landmarker model:

```text
models/pose_landmarker_full.task
```

The detector loads this model automatically when no custom model path is supplied. fileciteturn5file5L17-L44

The MediaPipe detector:

1. Receives an OpenCV BGR frame.
2. Converts the frame to RGB.
3. Converts it into a MediaPipe image.
4. Runs the Pose Landmarker.
5. Returns the pose detection result. fileciteturn5file5L70-L93

The detector is configured with:

```text
min_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5
num_poses = 1
```

by default. fileciteturn5file5L17-L23

### 3. Landmark Logger

Implemented in:

```text
src/landmark_logger.py
```

The logger records detected pose landmarks to:

```text
results/landmarks.csv
```

The output directory is automatically created if necessary. fileciteturn5file6L6-L19

Each detected landmark is recorded as one CSV row.

The CSV contains:

```text
timestamp
frame
landmark
x
y
z
visibility
presence
```

These fields are defined by the logger's CSV header. fileciteturn5file6L25-L45

## Main Integration Program

The main integration program is:

```text
main.py
```

It coordinates the camera, MediaPipe detector, landmark logger, and display functionality.

The overall processing flow is:

```text
Webcam
   │
   ▼
Camera
   │
   ▼
OpenCV Frame
   │
   ▼
MediaPipe Pose Landmarker
   │
   ▼
Pose Landmarks
   │
   ├───────────────┐
   │               │
   ▼               ▼
Display         CSV Logger
Landmarks       landmarks.csv
```

The integration program creates the camera, opens it, configures the requested resolution and FPS, creates the MediaPipe detector, and creates the landmark logger. fileciteturn5file1L100-L135

## Camera Configuration

The current requested webcam configuration is:

```text
Camera index: 0
Resolution: 1280 × 720
FPS: 30
```

These values are configured in `main.py`. fileciteturn5file1L8-L16

The program then applies the requested resolution and FPS to the webcam:

```text
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
FRAME_FPS    = 30
```

The values represent the **requested** camera settings; the actual values supported by a particular webcam may differ.

## Pose Landmark Visualisation

When MediaPipe detects a pose, `main.py`:

1. Takes the first detected pose.
2. Logs its landmarks.
3. Draws the pose landmarks.
4. Draws connections between selected landmarks.
5. Displays `PERSON DETECTED` on the video frame.

The implementation explicitly uses the first detected pose from the result. fileciteturn5file1L164-L185

The landmark visualisation converts MediaPipe's normalised landmark coordinates into pixel coordinates using the current frame width and height. fileciteturn5file1L61-L95

## Landmark Data

The MediaPipe landmarks contain positional and detection-related information.

The logger stores:

```text
x
y
z
visibility
presence
```

for each landmark. fileciteturn5file6L48-L86

The `x` and `y` values are normalised coordinates from the MediaPipe pose result, while `z` represents the landmark's depth-related coordinate provided by MediaPipe.

The logger does not calculate a fall decision. It records the detected landmark data for analysis and later processing.

## Live Processing Loop

The main loop continuously performs the following operations:

```text
Read webcam frame
       │
       ▼
Run MediaPipe Pose
       │
       ▼
Person detected?
   │           │
  YES          NO
   │            │
   ▼            ▼
Log landmarks  Display
Draw pose      "NO PERSON DETECTED"
Display status
       │
       ▼
Display live frame
       │
       ▼
Press Q?
   │
  YES
   │
   ▼
Release resources
```

The live processing loop reads frames, runs MediaPipe, logs detected landmarks, draws the pose, updates the status text, and displays the video. fileciteturn5file1L145-L211

The program can be terminated by pressing:

```text
Q
```

## Person Detection Behaviour

If MediaPipe detects at least one pose, the integration displays:

```text
PERSON DETECTED
```

If no pose is detected, it displays:

```text
NO PERSON DETECTED
```

The current implementation uses:

```python
result.pose_landmarks[0]
```

when a pose is detected, meaning the integration processes the **first detected pose**. fileciteturn5file1L164-L185

The underlying MediaPipe detector is configured with:

```text
num_poses = 1
```

by default. fileciteturn5file5L17-L23

## Landmark Logging

The logger is opened before the webcam processing loop begins:

```python
logger = LandmarkLogger(
    "results/landmarks.csv"
)

logger.open()
```

For every detected frame, the first detected pose is passed to:

```python
logger.log(
    frame_number,
    pose_landmarks
)
```

The logger writes all landmarks for that frame and flushes the file so the data is written to disk. fileciteturn5file1L164-L175 fileciteturn5file6L48-L90

## Resource Management

The integration program uses a `try/finally` block to release resources.

When the program ends, it:

1. Closes the landmark logger.
2. Closes the MediaPipe detector.
3. Releases the camera.
4. Destroys OpenCV windows.

This cleanup occurs even if the processing loop exits unexpectedly. fileciteturn5file1L215-L222

## Testing

The integration branch contains:

```text
tests/
├── test_camera.py
└── test_mediapipe.py
```

### Camera Tests

The camera tests verify:

- Live mode initialisation
- Default camera index
- Rejection of invalid modes
- Image mode requiring an image folder
- Live-mode status
- Image-mode status

These behaviours are directly tested in `test_camera.py`. fileciteturn5file3L7-L52

### MediaPipe Tests

The MediaPipe tests verify:

- The detector can initialise.
- A valid OpenCV image can be passed to the detector.
- Passing `None` raises a `ValueError`.
- `get_landmarks()` returns `None` when no person is detected in the synthetic blank image.

These tests are implemented in `test_mediapipe.py`. fileciteturn5file4L18-L56

Run the tests from the project root:

```bash
pytest -v
```

## Installation

Create and activate a virtual environment if required:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install the integration dependencies:

```bash
pip install -r requirements.txt
```

The current requirements are:

```text
mediapipe==1.0.1
opencv-python
numpy
pytest
```

fileciteturn5file2L1-L4

The MediaPipe model file must be present at:

```text
models/pose_landmarker_full.task
```

## Running the Integration Prototype

From the project root:

```bash
python main.py
```

The program starts the webcam and prints:

```text
======================================
ICT-304 MediaPipe Integration
======================================
Requested resolution: 1280x720
Requested FPS: 30
Press Q to quit.
```

The webcam window then displays the live camera feed with MediaPipe pose landmarks when a person is detected.

Press:

```text
Q
```

to stop the program.

## Generated Runtime Output

The integration generates:

```text
results/landmarks.csv
```

This file is intentionally excluded from version control by the branch's `.gitignore`:

```text
results/*.csv
```

along with virtual-environment files, Python cache files, compiled Python files, and MP4 files. fileciteturn5file0L1-L7

## Current System Scope

The current integration prototype demonstrates:

```text
Camera Input
      ↓
MediaPipe Pose Detection
      ↓
Pose Landmark Extraction
      ↓
Landmark Visualisation
      ↓
Landmark CSV Logging
```

This is an **integration prototype**, rather than the final complete fall-detection system.

In particular, the current `main.py` does not import or call the standalone `MediaPipeFallDetector` from the earlier MediaPipe branch. Instead, it detects and logs pose landmarks and displays the detection status. fileciteturn5file1L3-L5

Therefore, the current branch demonstrates successful integration of the **camera + MediaPipe Pose + landmark logging** components, while the fall-classification logic remains a subsequent integration step.

## Limitations

- The current integration uses one detected pose.
- `main.py` processes `result.pose_landmarks[0]`, so only the first detected pose is used. fileciteturn5file1L164-L175
- The current `main.py` does not perform torso-angle fall classification.
- The current prototype logs landmark data but does not generate a `FALL` / `NO_FALL` decision.
- The requested camera resolution and FPS may not be the actual values delivered by the webcam.
- The landmark CSV records every detected frame, which can produce a large file during extended runs.

## Future Integration Direction

The intended next stage is to connect the existing pose landmarks to the fall-detection logic.

The expanded processing pipeline would be:

```text
Webcam
   │
   ▼
Camera
   │
   ▼
OpenCV Frame
   │
   ▼
MediaPipe Pose
   │
   ▼
Pose Landmarks
   │
   ▼
Fall Detection Logic
   │
   ├── FALL
   │
   └── NO_FALL
          │
          ▼
   System Indicator
```

The existing integration structure already provides the camera frame and MediaPipe landmarks required for this next stage.

## Summary

The `feature/integration` branch combines the camera and MediaPipe components into a working live prototype.

The current system:

1. Opens a webcam.
2. Requests a 1280 × 720, 30 FPS configuration.
3. Captures live OpenCV frames.
4. Runs MediaPipe Pose detection.
5. Processes the first detected pose.
6. Draws pose landmarks and connections.
7. Displays whether a person is detected.
8. Logs landmark coordinates and detection information to CSV.
9. Provides automated tests for the camera and MediaPipe components.
10. Properly releases the camera, MediaPipe, logger, and OpenCV resources when the application exits.

The branch therefore represents the **system integration stage of the camera and MediaPipe components**, with fall-classification logic still to be incorporated into the final system.
