
from collections import deque
from pathlib import Path
from time import monotonic

import joblib
import numpy as np

from src.fall_features import (
    frame_features,
    temporal_features,
)


# --------------------------------------------------
# Random Forest V3 live fall detector
# --------------------------------------------------

class MLFallDetector:
    """
    Live fall detection using Random Forest V3.

    Uses the same 66 temporal features and alert
    settings as the V3 offline evaluator.

    This model predicts fall-like movement.
    It does not independently confirm a medical emergency.
    """

    PROBABILITY_THRESHOLD = 0.50
    EVIDENCE_WINDOW_SECONDS = 0.50
    MIN_POSITIVE_PREDICTIONS = 2
    ALERT_COOLDOWN_SECONDS = 8.0
    ALERT_DISPLAY_SECONDS = 3.0

    def __init__(self, model_path=None):
        if model_path is None:
            project_root = (
                Path(__file__).resolve().parents[1]
            )
            model_path = (
                project_root
                / "models"
                / "fall_classifier_v3.joblib"
            )

        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"V3 model not found: {self.model_path}"
            )

        saved = joblib.load(self.model_path)

        if saved.get("feature_version") != 1:
            raise ValueError(
                "Expected the V1/V3 66-feature model."
            )

        self.model = saved["model"]

        self.window_seconds = saved["window_seconds"]
        self.sample_fps = saved["sample_fps"]

        if self.model.n_features_in_ != 66:
            raise ValueError(
                "V3 model must use 66 input features."
            )

        self.history = deque()
        self.positive_times = deque()

        self.last_sample_time = float("-inf")
        self.last_alert_time = float("-inf")

        self.fall_probability = None
        self.predicted_fall = False

        self.alert_active_until = float("-inf")
        self.last_alert_triggered = False

    def reset(self):
        """Reset the detector's temporal state."""

        self.history.clear()
        self.positive_times.clear()

        self.last_sample_time = float("-inf")
        self.last_alert_time = float("-inf")

        self.fall_probability = None
        self.predicted_fall = False

        self.alert_active_until = float("-inf")
        self.last_alert_triggered = False

    def update(self, landmarks, timestamp=None):
        """
        Process the current frame's pose landmarks.

        Returns:
            fall_probability: float or None
            predicted_fall: bool
            alert_triggered: bool
            alert_active: bool
        """

        now = (
            monotonic()
            if timestamp is None
            else float(timestamp)
        )

        self.last_alert_triggered = False

        # Limit classification to the model's
        # sampling frequency.
        if (
            now - self.last_sample_time
            < 1.0 / self.sample_fps
        ):
            return self._result(now)

        self.last_sample_time = now

        # ----------------------------------
        # Extract pose features
        # ----------------------------------

        features = frame_features(landmarks)

        if features is None:
            self.history.clear()
            self.positive_times.clear()

            self.fall_probability = None
            self.predicted_fall = False

            return self._result(now)

        self.history.append((now, features))

        while (
            self.history
            and now - self.history[0][0]
            > self.window_seconds
        ):
            self.history.popleft()

        window = temporal_features(self.history)

        if window is None:
            self.fall_probability = None
            self.predicted_fall = False
            self.positive_times.clear()

            return self._result(now)

        # ----------------------------------
        # Random Forest prediction
        # ----------------------------------

        self.fall_probability = float(
            self.model.predict_proba(
                np.asarray(window).reshape(1, -1)
            )[0, 1]
        )

        self.predicted_fall = (
            self.fall_probability
            >= self.PROBABILITY_THRESHOLD
        )

        # ----------------------------------
        # Alert logic
        # ----------------------------------

        if self.predicted_fall:
            self.positive_times.append(now)

        while (
            self.positive_times
            and now - self.positive_times[0]
            > self.EVIDENCE_WINDOW_SECONDS
        ):
            self.positive_times.popleft()

        enough_evidence = (
            len(self.positive_times)
            >= self.MIN_POSITIVE_PREDICTIONS
        )

        cooldown_finished = (
            now - self.last_alert_time
            >= self.ALERT_COOLDOWN_SECONDS
        )

        if enough_evidence and cooldown_finished:
            self.last_alert_triggered = True
            self.last_alert_time = now

            self.alert_active_until = (
                now + self.ALERT_DISPLAY_SECONDS
            )

            # Require fresh positive predictions
            # before another alert.
            self.positive_times.clear()

        return self._result(now)

    def _result(self, now):
        return (
            self.fall_probability,
            self.predicted_fall,
            self.last_alert_triggered,
            now < self.alert_active_until,
        )
