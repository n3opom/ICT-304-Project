import cv2
import numpy as np
import pytest

from mediapipe.tasks.python.vision import PoseLandmark

from src.mediapipe import MediaPipePoseDetector
from src.mediapipe_fall_detection import (
    MediaPipeFallDetector
)


def create_test_image():
    """
    Create a valid OpenCV image.
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


def test_fall_detector_initialises():

    detector = MediaPipeFallDetector()

    assert detector is not None


def test_fall_detector_custom_parameters():

    detector = MediaPipeFallDetector(
        torso_angle_threshold=45.0,
        min_landmark_visibility=0.5
    )

    assert (
        detector.torso_angle_threshold
        == 45.0
    )

    assert (
        detector.min_landmark_visibility
        == 0.5
    )


def test_no_landmarks():

    detector = MediaPipeFallDetector()

    assert (
        detector.is_fallen(None)
        is False
    )

    assert (
        detector.classify(None)
        == "NO_PERSON"
    )


def test_vertical_pose_is_not_fall():

    class Landmark:

        def __init__(
            self,
            x,
            y,
            visibility=1.0
        ):
            self.x = x
            self.y = y
            self.visibility = visibility

    landmarks = [
        Landmark(0.0, 0.0)
        for _ in range(33)
    ]

    landmarks[
        PoseLandmark.LEFT_SHOULDER.value
    ] = Landmark(
        0.40,
        0.30
    )

    landmarks[
        PoseLandmark.RIGHT_SHOULDER.value
    ] = Landmark(
        0.60,
        0.30
    )

    landmarks[
        PoseLandmark.LEFT_HIP.value
    ] = Landmark(
        0.45,
        0.70
    )

    landmarks[
        PoseLandmark.RIGHT_HIP.value
    ] = Landmark(
        0.55,
        0.70
    )

    detector = MediaPipeFallDetector(
        torso_angle_threshold=60.0,
        min_landmark_visibility=0.5
    )

    assert (
        detector.is_fallen(landmarks)
        is False
    )


def test_horizontal_pose_is_fall():

    class Landmark:

        def __init__(
            self,
            x,
            y,
            visibility=1.0
        ):
            self.x = x
            self.y = y
            self.visibility = visibility

    landmarks = [
        Landmark(0.0, 0.0)
        for _ in range(33)
    ]

    landmarks[
        PoseLandmark.LEFT_SHOULDER.value
    ] = Landmark(
        0.30,
        0.50
    )

    landmarks[
        PoseLandmark.RIGHT_SHOULDER.value
    ] = Landmark(
        0.40,
        0.50
    )

    landmarks[
        PoseLandmark.LEFT_HIP.value
    ] = Landmark(
        0.70,
        0.55
    )

    landmarks[
        PoseLandmark.RIGHT_HIP.value
    ] = Landmark(
        0.80,
        0.55
    )

    detector = MediaPipeFallDetector(
        torso_angle_threshold=60.0,
        min_landmark_visibility=0.5
    )

    assert (
        detector.is_fallen(landmarks)
        is True
    )