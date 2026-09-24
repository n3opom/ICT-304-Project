import math

from mediapipe.tasks.python.vision import PoseLandmark


class MediaPipeFallDetector:
    """
    Determines whether a detected pose represents a fall.

    Fall detection is based on torso orientation.

    Approximately:

        0 degrees  = horizontal torso
        90 degrees = vertical torso
    """

    def __init__(
        self,
        torso_angle_threshold=60.0,
        min_landmark_visibility=0.5
    ):
        self.torso_angle_threshold = (
            torso_angle_threshold
        )

        self.min_landmark_visibility = (
            min_landmark_visibility
        )

    def _valid_landmark(self, landmark):
        """
        Check whether a landmark has sufficient visibility.
        """

        if landmark is None:
            return False

        visibility = landmark.visibility

        if visibility is None:
            return True

        return (
            visibility
            >= self.min_landmark_visibility
        )

    def _midpoint(self, point_a, point_b):
        """
        Calculate the midpoint between two landmarks.
        """

        return (
            (point_a.x + point_b.x) / 2.0,
            (point_a.y + point_b.y) / 2.0
        )

    def calculate_torso_angle(self, landmarks):
        """
        Calculate the torso angle relative to the horizontal axis.

        Returns:
            Angle in degrees.

        Approximately:

            0 degrees  = horizontal
            90 degrees = vertical

        Returns None if the required landmarks are
        unavailable or insufficiently visible.
        """

        left_shoulder = landmarks[
            PoseLandmark.LEFT_SHOULDER.value
        ]

        right_shoulder = landmarks[
            PoseLandmark.RIGHT_SHOULDER.value
        ]

        left_hip = landmarks[
            PoseLandmark.LEFT_HIP.value
        ]

        right_hip = landmarks[
            PoseLandmark.RIGHT_HIP.value
        ]

        required_landmarks = [
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip
        ]

        for landmark in required_landmarks:

            if not self._valid_landmark(
                landmark
            ):
                return None

        shoulder_center = self._midpoint(
            left_shoulder,
            right_shoulder
        )

        hip_center = self._midpoint(
            left_hip,
            right_hip
        )

        dx = (
            hip_center[0]
            - shoulder_center[0]
        )

        dy = (
            hip_center[1]
            - shoulder_center[1]
        )

        if dx == 0 and dy == 0:
            return None

        angle = math.degrees(
            math.atan2(
                abs(dy),
                abs(dx)
            )
        )

        return angle

    def is_fallen(self, landmarks):
        """
        Determine whether the pose is classified as a fall.
        """

        if landmarks is None:
            return False

        angle = self.calculate_torso_angle(
            landmarks
        )

        if angle is None:
            return False

        return (
            angle
            < self.torso_angle_threshold
        )

    def classify(self, landmarks):
        """
        Classify the detected pose.

        Returns:

            FALL
            NO_FALL
            NO_PERSON
        """

        if landmarks is None:
            return "NO_PERSON"

        if self.is_fallen(landmarks):
            return "FALL"

        return "NO_FALL"