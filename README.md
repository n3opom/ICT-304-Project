# ICT-304 AI System Design Project

## Project Overview

This repository contains the development work for the ICT-304 AI System Design project.

The project investigates computer-vision techniques for automated human fall detection using a camera. Two AI-based approaches were investigated independently:

- **YOLO object detection using bounding boxes**
- **MediaPipe Pose using body landmarks**

The project was developed incrementally using separate Git branches so that each major component and AI technique could be developed and tested independently before further integration work.

> **Important:** The `feature/integration` branch is a **technical integration prototype**, not the final system design. Its purpose was to verify that the camera, MediaPipe Pose, and related processing code could work together. The integration code is therefore different from the final intended fall-detection design.

---

## Project Objectives

The project aims to investigate and develop a computer-vision-based approach for detecting potential human falls.

The development process focuses on:

1. Establishing a reusable camera input component.
2. Investigating YOLO as a bounding-box-based detection technique.
3. Investigating MediaPipe Pose as a pose-landmark-based detection technique.
4. Testing the two techniques using labelled test images and parameter combinations.
5. Comparing the experimental results of the two approaches.
6. Creating a Camera + MediaPipe integration prototype to verify that the components can operate together.
7. Providing a foundation for the subsequent development of the final system design.

---

## AI Techniques Investigated

### YOLO Bounding Box Detection

The YOLO branch investigates fall detection using object detection and bounding-box geometry.

YOLO detects people in an image and provides bounding boxes around detected persons. The implementation then calculates the bounding-box width-to-height ratio and uses a configurable threshold to classify the detected person as either:

- `FALL`
- `NOT FALL`

The YOLO experiments tested combinations of detection confidence and bounding-box ratio thresholds.

The branch also records image-level and combination-level results, including:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- Accuracy
- Precision
- Recall
- F1-score

See the `feature/yolo` branch README for the detailed implementation and experimental procedure.

---

### MediaPipe Pose

The MediaPipe branch investigates human pose estimation using body landmarks.

Instead of representing a person using a rectangular bounding box, MediaPipe Pose provides landmark points representing different parts of the body. The implementation uses these landmarks to calculate body geometry, including torso orientation, which is then used as part of the fall-detection approach.

The MediaPipe branch also includes parameter testing and CSV-based experimental results for evaluating different combinations of fall-detection parameters.

See the `feature/mediapipe` branch README for the detailed implementation and experimental procedure.

---

## Repository Branches

The repository is organised into separate development branches.

| Branch | Purpose |
|---|---|
| `main` | Project-level repository and documentation |
| `feature/camera` | Development and testing of the camera input component |
| `feature/yolo` | YOLO-based fall-detection investigation |
| `feature/mediapipe` | MediaPipe Pose fall-detection investigation |
| `feature/integration` | Prototype used to verify that the computer-vision components can work together |

### Branch Development Approach

The branches were used to separate the development and testing of individual components.

The general development process was:

```text
Camera Development
       │
       ├───────────────┐
       │               │
       ▼               ▼
 YOLO Investigation   MediaPipe Investigation
       │               │
       └───────┬───────┘
               ▼
      Technique Comparison
               │
               ▼
       Camera + MediaPipe
       Integration Prototype
               │
               ▼
   Further Final-System Design
```

The integration branch should therefore **not be interpreted as the final implementation**. It was an intermediate technical check to determine whether the relevant components could operate together.

---

## Integration Prototype

The `feature/integration` branch combines the **Camera and MediaPipe Pose** components into a working prototype.

Its primary purpose was to verify that these two components could operate together. It was **not intended to implement the final fall-detection design**.

The prototype processing pipeline is:

```text
Webcam
   ↓
Camera Module
   ↓
OpenCV Frame
   ↓
MediaPipe Pose
   ↓
Pose Landmarks
   ↓
Landmark Logging
```

The current integration prototype:

- Opens the webcam through the Camera class.
- Captures OpenCV frames.
- Passes frames to MediaPipe Pose.
- Detects pose landmarks.
- Draws the detected pose on the video.
- Records landmark information into a CSV file.

The integration code is intentionally different from the individual YOLO and MediaPipe investigation code because it was developed as a separate prototype for checking component compatibility.

YOLO is **not part of this integration prototype**. The YOLO branch remains an independent investigation used for comparison with the MediaPipe approach.

The prototype is **not the final fall-detection implementation** and should not be treated as the final system architecture. The final system design may use different processing logic, classes, decision rules, and component interactions based on the investigation results and project requirements.

---

## Project Structure

The exact contents differ between branches. At a high level, the repository contains:

```text
ICT-304-Project/
│
├── data/
│   └── test/
│
├── src/
│   ├── camera.py
│   ├── yolo.py
│   ├── yolo_detection.py
│   ├── yolo_fall_detection.py
│   ├── mediapipe.py
│   └── ...
│
├── tests/
│   ├── test_camera.py
│   ├── test_yolo.py
│   ├── test_yolo_images.py
│   └── test_mediapipe.py
│
├── models/
│
├── results/
│
├── requirements.txt
│
└── README.md
```

Not every file shown above exists on every branch. Each feature branch contains the files relevant to its specific development stage.

---

## Testing

Testing was performed at both the component and experimental levels.

### Component Testing

Unit tests were created for components such as:

- Camera initialisation and operation
- Camera input modes
- YOLO functionality
- MediaPipe initialisation and image processing

The purpose of these tests was to verify that individual components behaved as expected before being used in larger processing pipelines.

### AI Technique Testing

The YOLO and MediaPipe branches also contain image-based testing.

The labelled test dataset contains fall and non-fall examples. The AI techniques were tested against these examples and their predictions were compared against the ground-truth labels.

Results were recorded in CSV files so that different parameter combinations could be compared.

---

## Experimental Results

The investigation produced separate results for the two AI techniques.

### YOLO

The YOLO testing evaluated combinations of:

- Detection confidence
- Bounding-box ratio threshold

The resulting CSV files contain raw detection information, image-level classifications, and summary metrics for parameter combinations.

### MediaPipe Pose

The MediaPipe testing evaluated combinations of parameters associated with the pose-based fall-detection method.

The resulting CSV files contain raw landmark/detection information, image-level classifications, and parameter-combination results.

The experimental results are used to understand the behaviour of each approach and inform the subsequent system-design decisions.

---

## Requirements

Each feature branch maintains its own requirements according to the libraries used by that branch.

Typical project dependencies include:

- Python
- OpenCV
- NumPy
- Pytest
- MediaPipe
- Ultralytics YOLO

Refer to the `requirements.txt` file on the specific branch being used before installing dependencies.

---

## Running the Project

Because the repository contains separate development branches, the appropriate branch should be selected depending on the component being tested.

For example:

```bash
git switch feature/camera
```

```bash
git switch feature/yolo
```

```bash
git switch feature/mediapipe
```

```bash
git switch feature/integration
```

Each branch contains its own implementation and documentation describing how that component is run.

---

## Development Workflow

The project uses Git for version control and separates major development activities into feature branches.

A typical workflow is:

```bash
git switch <branch>
git pull
```

After making changes:

```bash
git add .
git commit -m "Description of changes"
git push origin <branch>
```

The `main` branch serves as the project-level repository, while the feature branches preserve the development and testing work for the individual components.

---

## Current Development Status

The project has progressed through several stages:

1. **Camera component**
   - Camera input functionality developed and tested.

2. **YOLO investigation**
   - YOLO-based person detection and bounding-box fall classification investigated.
   - Parameter combinations tested and recorded.

3. **MediaPipe Pose investigation**
   - Pose landmark detection and pose-based fall classification investigated.
   - Parameter combinations tested and recorded.

4. **Technique comparison**
   - Results from the YOLO and MediaPipe investigations can be compared to support subsequent design decisions.

5. **Integration prototype**
   - Camera and MediaPipe processing were combined to verify that these components could operate together.
   - This branch is an intermediate technical prototype rather than the final system.
   - YOLO is not part of this integration prototype.

6. **Final system development**
   - The final system design and implementation can be developed based on the findings from the individual investigations and the requirements of the project.

---

## Important Repository Note

The different branches represent **different development stages and purposes**. Their code should not be expected to be identical.

In particular:

- The Camera branch focuses on reusable camera input.
- The YOLO branch focuses on object detection and bounding-box-based analysis.
- The MediaPipe branch focuses on pose landmarks and pose-based analysis.
- The Integration branch contains prototype code for testing Camera + MediaPipe component compatibility.
- The `main` branch provides the overall project documentation.

Therefore, differences between the branch implementations are intentional and reflect the development and testing process of the project.

---

## Project Documentation

Detailed documentation for each development stage is maintained within its respective branch:

- `feature/camera` — Camera component
- `feature/yolo` — YOLO investigation
- `feature/mediapipe` — MediaPipe Pose investigation
- `feature/integration` — Integration prototype

This README provides the overall project context, while the branch-specific READMEs provide implementation-level details.
