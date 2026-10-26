
from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import hypot
from time import monotonic

from mediapipe.tasks.python.vision import PoseLandmark


class Posture(str, Enum):
    STANDING = "STANDING"
    SITTING = "SITTING"
    LYING = "LYING"
    OTHER = "OTHER"
    UNCERTAIN = "UNCERTAIN"
    NO_PERSON = "NO_PERSON"


class FallState(str, Enum):
    NO_FALL = "NO FALL"
    POSSIBLE_FALL = "POSSIBLE FALL"
    FALL = "FALL"


@dataclass
class Observation:
    timestamp: float
    posture: Posture
    torso_angle: float | None
    body_center_y: float | None
    movement: float | None


class TemporalFallDetector:
    """
    Experimental temporal fall detector.

    Analyses successive pose observations instead of
    classifying a single horizontal pose as a fall.

    This is a rule-based prototype and requires testing
    with representative webcam recordings.
    """

    def __init__(
        self,
        min_landmark_visibility=0.5,
        history_seconds=3.0,
        rapid_transition_seconds=1.0,
        low_posture_seconds=1.0,
        possible_fall_angle=60.0,
        rapid_movement_threshold=0.08,
    ):
        self.min_landmark_visibility = (
            min_landmark_visibility
        )
        self.history_seconds = history_seconds
        self.rapid_transition_seconds = (
            rapid_transition_seconds
        )
        self.low_posture_seconds = low_posture_seconds
        self.possible_fall_angle = possible_fall_angle
        self.rapid_movement_threshold = (
            rapid_movement_threshold
        )

        self.history = deque()
        self.last_state = FallState.NO_FALL
        self.last_observation = None
        self.low_posture_start = None
        self.possible_fall_start = None

    def reset(self):
        """Clear temporal history and event state."""
        self.history.clear()
        self.last_state = FallState.NO_FALL
        self.last_observation = None
        self.low_posture_start = None
        self.possible_fall_start = None

    def _valid(self, landmark):
        if landmark is None:
            return False

        visibility = getattr(
            landmark, "visibility", None
        )

        return (
            visibility is None
            or visibility >= self.min_landmark_visibility
        )

    def _midpoint(self, a, b):
        return (
            (a.x + b.x) / 2.0,
            (a.y + b.y) / 2.0,
        )

    def _get_landmark(self, landmarks, name):
        return landmarks[getattr(PoseLandmark, name).value]

    def _features(self, landmarks):
        if landmarks is None:
            return None

        required_names = (
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_HIP",
            "RIGHT_HIP",
        )

        try:
            points = [
                self._get_landmark(landmarks, name)
                for name in required_names
            ]
        except (IndexError, AttributeError, TypeError):
            return None

        if not all(self._valid(p) for p in points):
            return None

        left_shoulder, right_shoulder, left_hip, right_hip = points

        shoulder = self._midpoint(
            left_shoulder, right_shoulder
        )
        hip = self._midpoint(left_hip, right_hip)

        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]

        if dx == 0 and dy == 0:
            return None

        from math import atan2, degrees

        angle = degrees(atan2(abs(dy), abs(dx)))

        # Use the torso midpoint as a relative image-space feature.
        center_y = (shoulder[1] + hip[1]) / 2.0

        return angle, center_y

    
    def _estimate_posture(self, landmarks, angle):
        """
        Estimate posture using torso orientation and leg geometry.

        This is a heuristic, not a trained posture classifier.
        Camera perspective and occlusion can make the result
        uncertain.
        """
        if landmarks is None or angle is None:
            return Posture.UNCERTAIN

        try:
            left_hip = self._get_landmark(
                landmarks, "LEFT_HIP"
            )
            right_hip = self._get_landmark(
                landmarks, "RIGHT_HIP"
            )
            left_knee = self._get_landmark(
                landmarks, "LEFT_KNEE"
            )
            right_knee = self._get_landmark(
                landmarks, "RIGHT_KNEE"
            )
            left_ankle = self._get_landmark(
                landmarks, "LEFT_ANKLE"
            )
            right_ankle = self._get_landmark(
                landmarks, "RIGHT_ANKLE"
            )
        except (IndexError, AttributeError, TypeError):
            return Posture.UNCERTAIN

        leg_landmarks = [
            left_hip,
            right_hip,
            left_knee,
            right_knee,
            left_ankle,
            right_ankle,
        ]

        # If the legs are not visible, do not guess the posture.
        if not all(self._valid(point) for point in leg_landmarks):
            return Posture.UNCERTAIN

        hip_y = (left_hip.y + right_hip.y) / 2.0
        knee_y = (left_knee.y + right_knee.y) / 2.0
        ankle_y = (left_ankle.y + right_ankle.y) / 2.0

        hip_to_knee = abs(knee_y - hip_y)
        knee_to_ankle = abs(ankle_y - knee_y)

        # A low torso angle suggests a horizontal or diagonal torso.
        # This alone does not prove a fall.
        if angle < self.possible_fall_angle:
            return Posture.LYING

        # Image coordinates increase downwards. A sitting person
        # often has knees closer to hip height than a standing person.
        if hip_to_knee < knee_to_ankle * 0.65:
            return Posture.SITTING

        return Posture.STANDING


    def update(self, landmarks, timestamp=None):
        """
        Process one frame's pose landmarks.

        Returns:
            (fall_state, posture)

        Supply monotonic timestamps in seconds when available.
        """
        now = monotonic() if timestamp is None else timestamp

        if landmarks is None:
            self.last_observation = None
            return FallState.NO_FALL, Posture.NO_PERSON

        features = self._features(landmarks)

        if features is None:
            return FallState.NO_FALL, Posture.UNCERTAIN

        angle, center_y = features
        posture = self._estimate_posture(
            landmarks, angle
        )

        movement = None

        if self.last_observation is not None:
            previous = self.last_observation
            elapsed = now - previous.timestamp

            if elapsed > 0:
                movement = abs(
                    center_y - previous.body_center_y
                )

        observation = Observation(
            timestamp=now,
            posture=posture,
            torso_angle=angle,
            body_center_y=center_y,
            movement=movement,
        )

        self.history.append(observation)
        self.last_observation = observation

        while (
            self.history
            and now - self.history[0].timestamp
            > self.history_seconds
        ):
            self.history.popleft()

        
        if posture == Posture.LYING:
            if self.low_posture_start is None:
                self.low_posture_start = now
        else:
            self.low_posture_start = None

        # Find the most recent upright observation before this frame.
        previous_upright = next(
            (
                item
                for item in reversed(list(self.history)[:-1])
                if item.torso_angle is not None
                and item.torso_angle >= self.possible_fall_angle
            ),
            None,
        )

        rapid_transition = (
            posture == Posture.LYING
            and previous_upright is not None
            and 0 < now - previous_upright.timestamp
            <= self.rapid_transition_seconds
        )

        rapid_movement = (
            movement is not None
            and movement >= self.rapid_movement_threshold
        )

        sustained_low_posture = (
            self.low_posture_start is not None
            and now - self.low_posture_start
            >= self.low_posture_seconds
        )

        # A rapid transition or sudden downward movement starts
        # a possible-fall candidate.
        if (
            posture == Posture.LYING
            and self.last_state == FallState.NO_FALL
            and (rapid_transition or rapid_movement)
        ):
            self.last_state = FallState.POSSIBLE_FALL
            self.possible_fall_start = now

        # Promote the candidate only after low posture persists.
        elif (
            posture == Posture.LYING
            and self.last_state == FallState.POSSIBLE_FALL
            and sustained_low_posture
        ):
            self.last_state = FallState.FALL

        # If the person returns upright or to another posture,
        # clear the candidate.
        elif posture not in (
            Posture.LYING,
            Posture.UNCERTAIN,
        ):
            self.last_state = FallState.NO_FALL
            self.possible_fall_start = None


        elif (
            self.last_state == FallState.POSSIBLE_FALL
            and sustained_low_posture
        ):
            self.last_state = FallState.FALL

        elif posture != Posture.LYING:
            self.last_state = FallState.NO_FALL
            self.possible_fall_start = None

        return self.last_state, posture
