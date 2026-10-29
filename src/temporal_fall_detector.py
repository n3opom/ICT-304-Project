from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import atan2, degrees
from statistics import median
from time import monotonic
from types import SimpleNamespace


class Posture(str, Enum):
    STANDING = "STANDING"
    SITTING = "SITTING"
    CROUCHING = "CROUCHING"
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
    """Experimental rule-based fall detector with coordinate stabilisation.

    Designed for a front-facing, full-body webcam view. The rules are heuristic
    and must be validated against labelled recordings before real-world use.
    """

    LANDMARK_INDEX = {
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
    }
    REQUIRED_INDICES = tuple(LANDMARK_INDEX.values())

    def __init__(
        self,
        min_landmark_visibility=0.5,
        history_seconds=3.0,
        rapid_transition_seconds=1.0,
        low_posture_seconds=1.0,
        possible_fall_angle=60.0,
        rapid_movement_threshold=0.08,
        tracking_loss_grace_seconds=0.5,
        unexplained_lying_seconds=3.0,
        fall_confirmation_seconds=None,
        unexplained_lying_confirmation_seconds=2.0,
    ):
        self.min_landmark_visibility = min_landmark_visibility
        self.history_seconds = history_seconds
        self.rapid_transition_seconds = rapid_transition_seconds
        self.low_posture_seconds = low_posture_seconds
        self.possible_fall_angle = possible_fall_angle
        self.rapid_movement_threshold = rapid_movement_threshold
        self.tracking_loss_grace_seconds = tracking_loss_grace_seconds
        self.unexplained_lying_seconds = unexplained_lying_seconds
        self.fall_confirmation_seconds = (
            low_posture_seconds if fall_confirmation_seconds is None
            else fall_confirmation_seconds
        )
        # Retained for constructor compatibility; unexplained lying is not
        # automatically promoted to FALL based on duration alone.
        self.unexplained_lying_confirmation_seconds = unexplained_lying_confirmation_seconds

        self.rapid_angle_change_threshold = 90.0
        self.recovery_seconds = 0.6
        self.max_posture_uncertainty_seconds = 0.6
        self.candidate_timeout_seconds = 3.0

        self.history = deque()
        self.raw_coordinate_history = deque(maxlen=3)
        self.filtered_landmarks = None
        self.last_state = FallState.NO_FALL
        self.last_observation = None
        self.low_posture_start = None
        self.possible_fall_start = None
        self.candidate_from_lying_only = False
        self.recovery_start = None
        self.tracking_lost_start = None
        self.posture_uncertain_start = None
        self.last_upright_timestamp = None
        self.last_upright_angle = None
        self.controlled_lying_start = None
        self.last_filter_timestamp = None

    def reset(self):
        """Reset temporal evidence and smoothing history."""
        self.history.clear()
        self.raw_coordinate_history.clear()
        self.filtered_landmarks = None
        self.last_state = FallState.NO_FALL
        self.last_observation = None
        self.low_posture_start = None
        self.possible_fall_start = None
        self.candidate_from_lying_only = False
        self.recovery_start = None
        self.tracking_lost_start = None
        self.posture_uncertain_start = None
        self.last_upright_timestamp = None
        self.last_upright_angle = None
        self.controlled_lying_start = None
        self.last_filter_timestamp = None

    def _valid(self, landmark):
        if landmark is None:
            return False
        visibility = getattr(landmark, "visibility", None)
        return visibility is None or visibility >= self.min_landmark_visibility

    def _get_landmark(self, landmarks, name):
        try:
            return landmarks[self.LANDMARK_INDEX[name]]
        except (IndexError, KeyError, TypeError):
            return None

    @staticmethod
    def _midpoint(a, b):
        return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

    def _smooth_landmarks(self, landmarks):
        """Median-filter coordinate spikes over 3 frames, preserving metadata.

        A short median window rejects isolated landmark jumps while adding less
        delay than a long moving average. The returned landmarks are copies;
        MediaPipe's raw result is never modified.
        """
        self.raw_coordinate_history.append([
            (float(p.x), float(p.y), float(getattr(p, "z", 0.0)))
            for p in landmarks
        ])
        filtered = []
        # Do not average the first two frames: that would delay a genuine
        # rapid transition before the filter has enough samples to reject spikes.
        enough_samples = len(self.raw_coordinate_history) == self.raw_coordinate_history.maxlen
        for i, point in enumerate(landmarks):
            xs = [frame[i][0] for frame in self.raw_coordinate_history]
            ys = [frame[i][1] for frame in self.raw_coordinate_history]
            zs = [frame[i][2] for frame in self.raw_coordinate_history]
            filtered.append(SimpleNamespace(
                x=median(xs) if enough_samples else float(point.x),
                y=median(ys) if enough_samples else float(point.y),
                z=median(zs) if enough_samples else float(getattr(point, "z", 0.0)),
                visibility=getattr(point, "visibility", None),
                presence=getattr(point, "presence", None),
            ))
        return filtered

    def _features(self, landmarks):
        if landmarks is None:
            return None
        names = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")
        points = [self._get_landmark(landmarks, n) for n in names]
        if not all(self._valid(p) for p in points):
            return None
        left_shoulder, right_shoulder, left_hip, right_hip = points
        shoulder = self._midpoint(left_shoulder, right_shoulder)
        hip = self._midpoint(left_hip, right_hip)
        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        angle = degrees(atan2(abs(dy), abs(dx)))
        center_y = (shoulder[1] + hip[1]) / 2.0
        center_x = (shoulder[0] + hip[0]) / 2.0
        return angle, center_y, center_x

    def _estimate_posture(self, landmarks, angle):
        """Classify from reliable torso and available whole-body geometry.

        Leg occlusion must not automatically suppress a horizontal torso.
        A near-horizontal body may also be inferred from head-to-ankle or
        shoulder-to-knee alignment when the torso tracker is unreliable.
        """
        if landmarks is None or angle is None:
            return Posture.UNCERTAIN

        def midpoint(indices, minimum=1):
            pts = [landmarks[i] for i in indices if i < len(landmarks)
                   and self._valid(landmarks[i])]
            if len(pts) < minimum:
                return None
            return (sum(p.x for p in pts) / len(pts),
                    sum(p.y for p in pts) / len(pts))

        shoulder = midpoint((11, 12))
        hip = midpoint((23, 24))
        knee = midpoint((25, 26))
        ankle = midpoint((27, 28))
        nose = midpoint((0,))
        if shoulder is None or hip is None:
            return Posture.UNCERTAIN

        # Horizontal torso is strong evidence regardless of leg visibility.
        if angle <= 40.0:
            return Posture.LYING

        # Evaluate the visible body axis, not just the torso. A horizontal
        # head-to-feet arrangement is a lying cue even if the torso is skewed.
        points = [p for p in (nose, shoulder, hip, knee, ankle) if p is not None]
        if len(points) >= 3:
            span_x = max(p[0] for p in points) - min(p[0] for p in points)
            span_y = max(p[1] for p in points) - min(p[1] for p in points)
            if span_x >= 0.28 and span_x >= span_y * 1.25:
                return Posture.LYING

        # Without legs, a vertical torso cannot distinguish standing, sitting
        # or crouching. Avoid inventing a specific posture.
        if knee is None or ankle is None:
            return Posture.UNCERTAIN

        hip_knee_y = abs(knee[1] - hip[1])
        knee_ankle_y = abs(ankle[1] - knee[1])
        leg_vertical = ankle[1] - hip[1]
        leg_horizontal = abs(ankle[0] - hip[0])

        # When the projected legs are mostly horizontal and sufficiently long,
        # prefer lying over a seated/crouched label.
        if leg_horizontal >= 0.22 and leg_horizontal >= abs(leg_vertical) * 1.25:
            return Posture.LYING

        if angle < 65.0:
            return Posture.CROUCHING if hip_knee_y < 0.12 else Posture.UNCERTAIN
        if knee_ankle_y < 0.015 or hip_knee_y < 0.015:
            return Posture.UNCERTAIN
        if hip_knee_y < knee_ankle_y * 0.55:
            return Posture.SITTING
        if hip_knee_y < knee_ankle_y * 0.80:
            return Posture.CROUCHING
        return Posture.STANDING

    def _handle_tracking_loss(self, now, posture):
        if self.tracking_lost_start is None:
            self.tracking_lost_start = now
        self.last_observation = None
        self.raw_coordinate_history.clear()
        self.filtered_landmarks = None
        if now - self.tracking_lost_start > self.tracking_loss_grace_seconds:
            # Preserve the interval start while resetting the other state.
            lost_since = self.tracking_lost_start
            self.reset()
            self.tracking_lost_start = lost_since
            return FallState.NO_FALL, posture
        return self.last_state, posture

    def update(self, landmarks, timestamp=None):
        """Process one frame; returns (fall_state, posture)."""
        now = monotonic() if timestamp is None else float(timestamp)
        if landmarks is None:
            return self._handle_tracking_loss(now, Posture.NO_PERSON)

        # Required torso landmarks must be valid before accepting a frame.
        raw_features = self._features(landmarks)
        if raw_features is None:
            return self._handle_tracking_loss(now, Posture.UNCERTAIN)

        if self.tracking_lost_start is not None:
            gap = now - self.tracking_lost_start
            if gap > self.tracking_loss_grace_seconds:
                self.reset()
            else:
                # Keep alert state, but don't compute motion across the gap or
                # carry a pre-gap coordinate filter into the resumed frame.
                self.last_observation = None
                self.raw_coordinate_history.clear()
            self.tracking_lost_start = None

        # If frames arrive with a large gap, don't let old coordinates dominate
        # the current pose; rebuild the filter from the current frame.
        if self.last_filter_timestamp is not None and now - self.last_filter_timestamp > 0.25:
            self.raw_coordinate_history.clear()
        filtered = self._smooth_landmarks(landmarks)
        self.last_filter_timestamp = now
        self.filtered_landmarks = filtered
        features = self._features(filtered)
        if features is None:
            return self._handle_tracking_loss(now, Posture.UNCERTAIN)
        angle, center_y, _center_x = features
        posture = self._estimate_posture(filtered, angle)

        movement = None
        angle_change_rate = None
        previous = self.last_observation
        if previous is not None:
            elapsed = now - previous.timestamp
            if elapsed > 0:
                movement = abs(center_y - previous.body_center_y)
                if previous.torso_angle is not None:
                    angle_change_rate = abs(angle - previous.torso_angle) / elapsed

        observation = Observation(now, posture, angle, center_y, movement)
        self.history.append(observation)
        self.last_observation = observation
        while self.history and now - self.history[0].timestamp > self.history_seconds:
            self.history.popleft()

        if posture in (Posture.STANDING, Posture.SITTING, Posture.CROUCHING):
            self.last_upright_timestamp = now
            self.last_upright_angle = angle

        previous_upright = next((
            item for item in reversed(list(self.history)[:-1])
            if item.posture in (Posture.STANDING, Posture.SITTING, Posture.CROUCHING)
            and item.torso_angle is not None
            and 0 < now - item.timestamp <= self.rapid_transition_seconds
        ), None)
        rapid_transition = (
            posture == Posture.LYING
            and previous_upright is not None
            and angle_change_rate is not None
            and angle_change_rate >= self.rapid_angle_change_threshold
        )

        # Maintain a lying timer, tolerating a short uncertain interval.
        if posture == Posture.LYING:
            self.posture_uncertain_start = None
            if self.low_posture_start is None:
                self.low_posture_start = now
        elif posture == Posture.UNCERTAIN:
            if self.posture_uncertain_start is None:
                self.posture_uncertain_start = now
            elif now - self.posture_uncertain_start > self.max_posture_uncertainty_seconds:
                self.low_posture_start = None
        else:
            self.posture_uncertain_start = None
            self.low_posture_start = None
            self.controlled_lying_start = None

        sustained_lying = (
            self.low_posture_start is not None
            and now - self.low_posture_start >= self.low_posture_seconds
        )

        # Path 1: a rapid upright-to-lying transition creates a candidate.
        if posture == Posture.LYING and self.last_state == FallState.NO_FALL and rapid_transition:
            self.last_state = FallState.POSSIBLE_FALL
            self.possible_fall_start = now
            self.candidate_from_lying_only = False
            self.recovery_start = None
            self.controlled_lying_start = None

        # Path 2: if the system first observes a person lying, or did not see
        # a clear transition, raise only POSSIBLE FALL after a delay. Duration
        # alone never confirms FALL. If an upright pose was recently seen and
        # the transition was gradual, treat it as controlled lying instead.
        elif (
            posture == Posture.LYING
            and self.last_state == FallState.NO_FALL
            and sustained_lying
            and self.low_posture_start is not None
            and now - self.low_posture_start >= self.unexplained_lying_seconds
        ):
            upright_recently = (
                self.last_upright_timestamp is not None
                and self.low_posture_start - self.last_upright_timestamp <= self.history_seconds
            )
            if not upright_recently:
                self.last_state = FallState.POSSIBLE_FALL
                self.possible_fall_start = now
                self.candidate_from_lying_only = True
                self.recovery_start = None

        # Transition-based candidates need reliable sustained lying evidence.
        elif self.last_state == FallState.POSSIBLE_FALL and not self.candidate_from_lying_only:
            if posture == Posture.LYING and sustained_lying and self.possible_fall_start is not None:
                if now - self.possible_fall_start >= self.fall_confirmation_seconds:
                    self.last_state = FallState.FALL
                    self.recovery_start = None
            elif (
                self.possible_fall_start is not None
                and now - self.possible_fall_start > self.candidate_timeout_seconds
                and posture != Posture.LYING
            ):
                self.last_state = FallState.NO_FALL
                self.possible_fall_start = None
                self.candidate_from_lying_only = False

        # A person returning to a clearly non-lying posture cancels a candidate.
        if self.last_state == FallState.POSSIBLE_FALL and posture == Posture.STANDING:
            self.last_state = FallState.NO_FALL
            self.possible_fall_start = None
            self.candidate_from_lying_only = False
            self.recovery_start = None
            self.low_posture_start = None

        # Clear confirmed FALL only after stable recovery.
        elif self.last_state == FallState.FALL:
            if posture in (Posture.STANDING, Posture.SITTING):
                if self.recovery_start is None:
                    self.recovery_start = now
                elif now - self.recovery_start >= self.recovery_seconds:
                    self.last_state = FallState.NO_FALL
                    self.possible_fall_start = None
                    self.candidate_from_lying_only = False
                    self.recovery_start = None
                    self.low_posture_start = None
            else:
                self.recovery_start = None

        return self.last_state, posture
