from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipePoseDetector:
    """
    MediaPipe Pose Landmarker detector.

    Uses the current MediaPipe Tasks API and is compatible
    with modern MediaPipe releases such as MediaPipe 1.0.1.
    """

    def __init__(
        self,
        model_path=None,
        min_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1
    ):
        if model_path is None:
            project_root = Path(__file__).resolve().parents[1]

            model_path = (
                project_root
                / "pose_landmarker_full.task"
            )

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                "MediaPipe pose model was not found:\n"
                f"{model_path}\n\n"
                "Place pose_landmarker_full.task inside "
                "the project's models folder."
            )

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=(
                min_detection_confidence
            ),
            min_pose_presence_confidence=(
                min_pose_presence_confidence
            ),
            min_tracking_confidence=(
                min_tracking_confidence
            ),
            output_segmentation_masks=False
        )

        self.landmarker = (
            vision.PoseLandmarker.create_from_options(
                options
            )
        )

    def detect(self, image):
        """
        Detect pose landmarks from an OpenCV BGR image.

        Returns:
            MediaPipe PoseLandmarkerResult.
        """

        if image is None:
            raise ValueError(
                "Image cannot be None."
            )

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        return self.landmarker.detect(
            mp_image
        )

    def get_landmarks(self, image):
        """
        Detect a person and return the landmarks
        belonging to the first detected pose.

        Returns:
            List of landmarks, or None.
        """

        result = self.detect(image)

        if not result.pose_landmarks:
            return None

        return result.pose_landmarks[0]

    def has_person(self, image):
        """
        Return True if at least one person is detected.
        """

        result = self.detect(image)

        return bool(result.pose_landmarks)

    def close(self):
        """
        Release MediaPipe resources.
        """

        self.landmarker.close()