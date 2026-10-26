
import pytest
from types import SimpleNamespace

from src.temporal_fall_detector import (
    TemporalFallDetector,
    FallState,
    Posture,
)



def make_landmarks(
    torso_angle,
    center_y=0.5,
    visibility=0.99,
    posture="standing",
):
    """
    Build simplified synthetic MediaPipe-style landmarks.

    These landmarks are designed for logic tests, not to
    reproduce actual MediaPipe predictions.
    """
    import math

    landmarks = [
        SimpleNamespace(x=0.5, y=0.5, visibility=visibility)
        for _ in range(33)
    ]

    angle = math.radians(torso_angle)

    # Shoulder and hip positions define torso orientation.
    shoulder_x = 0.5 - 0.1 * math.cos(angle)
    shoulder_y = center_y - 0.1 * math.sin(angle)
    hip_x = 0.5 + 0.1 * math.cos(angle)
    hip_y = center_y + 0.1 * math.sin(angle)

    # Default leg geometry: standing.
    if posture == "sitting":
        knee_y = hip_y + 0.05
        ankle_y = knee_y + 0.20
    elif posture == "lying":
        knee_y = hip_y + 0.02
        ankle_y = knee_y + 0.02
    else:
        knee_y = hip_y + 0.20
        ankle_y = knee_y + 0.20

    # Shoulders
    for index in (11, 12):
        landmarks[index] = SimpleNamespace(
            x=shoulder_x,
            y=shoulder_y,
            visibility=visibility,
        )

    # Hips
    for index in (23, 24):
        landmarks[index] = SimpleNamespace(
            x=hip_x,
            y=hip_y,
            visibility=visibility,
        )

    # Knees
    for index in (25, 26):
        landmarks[index] = SimpleNamespace(
            x=hip_x,
            y=knee_y,
            visibility=visibility,
        )

    # Ankles
    for index in (27, 28):
        landmarks[index] = SimpleNamespace(
            x=hip_x,
            y=ankle_y,
            visibility=visibility,
        )

    return landmarks



def test_no_person_returns_no_person():
    detector = TemporalFallDetector()

    state, posture = detector.update(
        None,
        timestamp=0.0,
    )

    assert state == FallState.NO_FALL
    assert posture == Posture.NO_PERSON


def test_vertical_torso_is_not_lying():
    detector = TemporalFallDetector()

    landmarks = make_landmarks(90)

    state, posture = detector.update(
        landmarks,
        timestamp=0.0,
    )

    assert posture == Posture.STANDING
    assert state == FallState.NO_FALL


def test_horizontal_torso_is_initially_possible_fall():
    detector = TemporalFallDetector()

    # First observe an upright torso.
    detector.update(
        make_landmarks(90, center_y=0.4),
        timestamp=0.0,
    )

    # Then observe a horizontal torso shortly afterwards.
    state, posture = detector.update(
        make_landmarks(20, center_y=0.7),
        timestamp=0.2,
    )

    assert posture == Posture.LYING
    assert state == FallState.POSSIBLE_FALL


def test_missing_landmarks_are_uncertain():
    detector = TemporalFallDetector()

    landmarks = make_landmarks(90)

    # Hide one required landmark.
    landmarks[11].visibility = 0.1

    state, posture = detector.update(
        landmarks,
        timestamp=0.0,
    )

    assert state == FallState.NO_FALL
    assert posture == Posture.UNCERTAIN


def test_reset_clears_history():
    detector = TemporalFallDetector()

    detector.update(
        make_landmarks(90),
        timestamp=0.0,
    )

    detector.reset()

    assert len(detector.history) == 0
    assert detector.last_observation is None
    assert detector.last_state == FallState.NO_FALL


def test_horizontal_posture_can_persist_as_fall():
    detector = TemporalFallDetector(
        low_posture_seconds=0.5,
    )

    detector.update(
        make_landmarks(90),
        timestamp=0.0,
    )

    detector.update(
        make_landmarks(20, center_y=0.7),
        timestamp=0.2,
    )

    state, posture = detector.update(
        make_landmarks(20, center_y=0.7),
        timestamp=0.8,
    )

    assert posture == Posture.LYING
    assert state == FallState.FALL


def test_standing_posture_is_classified_correctly():
    detector = TemporalFallDetector()

    landmarks = make_landmarks(
        90,
        posture="standing",
    )

    _, posture = detector.update(
        landmarks,
        timestamp=0.0,
    )

    assert posture == Posture.STANDING


def test_sitting_posture_is_classified_correctly():
    detector = TemporalFallDetector()

    landmarks = make_landmarks(
        90,
        posture="sitting",
    )

    _, posture = detector.update(
        landmarks,
        timestamp=0.0,
    )

    assert posture == Posture.SITTING


def test_lying_posture_is_classified_correctly():
    detector = TemporalFallDetector()

    landmarks = make_landmarks(
        0,
        posture="lying",
    )

    _, posture = detector.update(
        landmarks,
        timestamp=0.0,
    )

    assert posture == Posture.LYING


def test_low_visibility_leg_landmarks_are_uncertain():
    detector = TemporalFallDetector()

    landmarks = make_landmarks(
        90,
        posture="standing",
    )

    # Make both knee landmarks unreliable.
    landmarks[25].visibility = 0.1
    landmarks[26].visibility = 0.1

    _, posture = detector.update(
        landmarks,
        timestamp=0.0,
    )

    assert posture == Posture.UNCERTAIN

def test_gradual_lying_down_does_not_trigger_fall():
    detector = TemporalFallDetector(
        rapid_transition_seconds=1.0,
        low_posture_seconds=1.0,
    )

    # The person is initially upright.
    detector.update(
        make_landmarks(90, posture="standing"),
        timestamp=0.0,
    )

    # The person remains upright for several seconds.
    detector.update(
        make_landmarks(90, posture="standing"),
        timestamp=2.0,
    )

    # The person is now lying down, but the transition
    # from the last upright observation took too long.
    state, posture = detector.update(
        make_landmarks(0, posture="lying"),
        timestamp=4.0,
    )

    assert posture == Posture.LYING
    assert state == FallState.NO_FALL

def test_rapid_transition_followed_by_lying_triggers_fall():
    detector = TemporalFallDetector(
        rapid_transition_seconds=1.0,
        low_posture_seconds=1.0,
    )

    # Start upright.
    detector.update(
        make_landmarks(90, posture="standing"),
        timestamp=0.0,
    )

    # Rapid transition to a horizontal torso.
    state, posture = detector.update(
        make_landmarks(0, posture="lying"),
        timestamp=0.2,
    )

    assert posture == Posture.LYING
    assert state == FallState.POSSIBLE_FALL

    # The person remains lying down.
    state, posture = detector.update(
        make_landmarks(0, posture="lying"),
        timestamp=1.3,
    )

    assert posture == Posture.LYING
    assert state == FallState.FALL


def test_rapid_movement_while_standing_does_not_trigger_fall():
    detector = TemporalFallDetector()

    # Initial upright position.
    detector.update(
        make_landmarks(
            90,
            center_y=0.4,
            posture="standing",
        ),
        timestamp=0.0,
    )

    # The torso moves vertically, but remains upright.
    state, posture = detector.update(
        make_landmarks(
            90,
            center_y=0.6,
            posture="standing",
        ),
        timestamp=0.2,
    )

    assert posture == Posture.STANDING
    assert state == FallState.NO_FALL


def test_returning_upright_clears_possible_fall():
    detector = TemporalFallDetector(
        rapid_transition_seconds=1.0,
        low_posture_seconds=1.0,
    )

    detector.update(
        make_landmarks(90, posture="standing"),
        timestamp=0.0,
    )

    state, _ = detector.update(
        make_landmarks(0, posture="lying"),
        timestamp=0.2,
    )
    assert state == FallState.POSSIBLE_FALL

    state, posture = detector.update(
        make_landmarks(90, posture="standing"),
        timestamp=0.5,
    )

    assert posture == Posture.STANDING
    assert state == FallState.NO_FALL
