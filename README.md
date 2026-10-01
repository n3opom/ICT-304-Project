# ICT-304 Project — Camera Branch

## Overview

The `feature/camera` branch provides the camera/input component for the ICT-304 AI System Design project.

The purpose of this branch is to provide a common input interface that can supply image frames from either:

1. **LIVE mode** — a connected webcam.
2. **IMAGE mode** — pre-taken test images stored in `data/test`.

The `Camera` class returns frames as OpenCV images so that the frames can be passed to an AI detection technique in later branches or during the integration phase.

The camera component itself does **not** determine whether a person has fallen. It is responsible for obtaining and supplying image frames to the AI detection system.

## Branch Structure

```text
ICT-304-Project/
│
├── .vscode/
│
├── data/
│   └── test/
│       ├── fall###.jpg
│       └── notfallen###.jpg
│
├── src/
│   ├── __init__.py
│   └── camera.py
│
├── tests/
│   └── test_camera.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Test Dataset

The `data/test` directory contains labelled JPG test images.

The filenames indicate the ground-truth category:

- `fall###.jpg` — image labelled as a **fall**.
- `notfallen###.jpg` — image labelled as **not fallen**.

The labels are represented by the filenames and are intended to support testing and evaluation of the AI fall-detection techniques. The camera module itself only loads and returns the images; it does not interpret the labels.

## Camera Module

The main implementation is located in:

```text
src/camera.py
```

The `Camera` class supports two input modes:

### 1. Live Webcam Mode

```python
camera = Camera(mode="live")
camera.open()

frame = camera.read()

camera.release()
```

In live mode, OpenCV's `VideoCapture` is used to access the selected webcam. The default camera index is `0`.

The camera can also be run directly from `camera.py`, where the user can select:

```text
1. Live webcam
2. Test images
```

In live mode, pressing `Q` exits the camera display.

### 2. Test Image Mode

```python
camera = Camera(
    mode="image",
    image_folder="data/test"
)

camera.open()

frame = camera.read()

camera.release()
```

In image mode, the class:

- checks that the supplied folder exists;
- finds `.jpg`, `.jpeg`, and `.png` files;
- sorts the image filenames;
- loads one image at a time;
- returns `None` after all images have been read.

This allows the same camera interface to be used with either a live webcam or a fixed test dataset.

## Main Camera Functions

### `open()`

Initialises the selected input source.

For live mode, it opens the webcam.

For image mode, it validates the test-image directory and loads the available image filenames.

### `read()`

Returns the next OpenCV frame.

- In live mode, it returns the next webcam frame.
- In image mode, it returns the next test image.
- In image mode, `None` is returned after all images have been processed.

### `release()`

Releases the webcam resource when applicable.

### `is_live()`

Returns whether the camera is currently operating in live mode.

### `show_frame()`

Displays an OpenCV frame in a window.

## Installation

Create and activate a Python virtual environment if required:

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

The current requirements are:

```text
opencv-python
pytest
```

## Running the Camera

From the project root:

```bash
python src/camera.py
```

The program will display:

```text
Camera Test
-----------
1. Live webcam
2. Test images
```

### Live webcam

Enter:

```text
1
```

The webcam will open. Press `Q` to quit.

### Test images

Enter:

```text
2
```

The images in:

```text
data/test
```

will be displayed one at a time.

Press any key to advance to the next image, or press `Q` to quit.

## Testing

The automated camera tests are located in:

```text
tests/test_camera.py
```

Run the complete test suite from the project root:

```bash
pytest -v
```

The tests verify that:

- the `data/test` directory exists;
- test images are present;
- the image-mode camera can be opened;
- an image can be read successfully;
- all available test images can be read;
- the camera returns `None` after all test images have been processed.

The automated tests focus on the **IMAGE mode**, allowing the camera input component to be tested without requiring a physical webcam during automated testing.

## Role in the Overall AI System

The camera branch acts as the **input layer** of the fall-detection system.

The intended processing flow is:

```text
Camera Input
     │
     ├── Live Webcam
     │
     └── Test Images
             │
             ▼
       OpenCV Frame
             │
             ▼
      AI Detection Model
             │
             ▼
       Fall Detection
             │
             ▼
     System Response
```

During integration, the same `Camera` interface can therefore be used to provide frames to the selected AI technique, such as MediaPipe Pose or YOLO.

This separation allows the camera/input functionality to be developed and tested independently from the AI detection implementation.

## Relationship to the Test Images

The labelled images in `data/test` are used as fixed input data for development and testing.

For example:

```text
data/test/
├── fall008.jpg
├── fall009.jpg
├── fall010.jpg
├── notfallen001.jpg
├── notfallen002.jpg
└── ...
```

The filename provides the expected ground-truth category. The camera component does not use this filename to make a prediction; the filename is available for the testing and evaluation stages of the fall-detection system.

## Notes

- Supported image formats are `.jpg`, `.jpeg`, and `.png`.
- Image files are processed in sorted filename order.
- The default webcam index is `0`.
- The camera module requires OpenCV.
- Automated tests use the test-image mode rather than relying on a physical webcam.
- The `.gitignore` excludes the virtual environment, Python cache files, compiled Python files, and MP4 files.
