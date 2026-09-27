import cv2
import numpy as np
import pytest

from src.mediapipe import MediaPipePoseDetector


def create_test_image():
    """
    Create a valid OpenCV image for testing.
    """
    return np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )


def test_mediapipe_detector_initialises():
    detector = MediaPipePoseDetector()

    assert detector is not None

    detector.close()


def test_mediapipe_detector_accepts_image():
    detector = MediaPipePoseDetector()

    image = create_test_image()

    result = detector.detect(image)

    assert result is not None

    detector.close()


def test_mediapipe_empty_image_raises_error():
    detector = MediaPipePoseDetector()

    with pytest.raises(ValueError):
        detector.detect(None)

    detector.close()


def test_mediapipe_get_landmarks_returns_none_when_no_person():
    detector = MediaPipePoseDetector()

    image = create_test_image()

    landmarks = detector.get_landmarks(image)

    assert landmarks is None

    detector.close()