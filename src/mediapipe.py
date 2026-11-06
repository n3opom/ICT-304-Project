
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipePoseDetector:
    """Detect and track human poses in consecutive frames."""

    def __init__(
        self,
        model_path=None,
        min_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    ):
        if model_path is None:
            project_root = Path(__file__).resolve().parents[1]
            model_path = (
                project_root / "models" / "pose_landmarker_full.task"
            )

        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {self.model_path}"
            )

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(
            options
        )
        self._last_timestamp_ms = -1

    def detect(self, image, timestamp_ms=None):
        """
        Detect poses in a BGR image.

        If timestamp_ms is provided, it must be greater than the
        timestamp used for the previous frame. If omitted, a
        monotonically increasing timestamp is generated automatically.
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        if not hasattr(image, "shape") or len(image.shape) != 3:
            raise ValueError("Input must be a valid BGR image.")

        if image.shape[2] != 3:
            raise ValueError("Input image must have 3 color channels.")

        if timestamp_ms is None:
            timestamp_ms = time.monotonic_ns() // 1_000_000
            timestamp_ms = max(
                timestamp_ms, self._last_timestamp_ms + 1
            )
        elif (
            not isinstance(timestamp_ms, int)
            or timestamp_ms <= self._last_timestamp_ms
        ):
            raise ValueError(
                "timestamp_ms must be an integer greater than "
                "the previous frame timestamp."
            )

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        result = self.landmarker.detect_for_video(
            mp_image, timestamp_ms
        )
        self._last_timestamp_ms = timestamp_ms

        return result

    def get_landmarks(self, image):
        """Return the first detected person's pose landmarks, or None."""
        result = self.detect(image)

        if result.pose_landmarks:
            return result.pose_landmarks[0]

        return None

    def has_person(self, image):
        """Return whether a person with detectable pose landmarks exists."""
        return self.get_landmarks(image) is not None

    def close(self):
        """Release MediaPipe resources."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
