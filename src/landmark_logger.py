import csv
from datetime import datetime
from pathlib import Path


class LandmarkLogger:
    """Log raw and filtered MediaPipe landmarks plus detector diagnostics."""

    def __init__(self, output_path="results/landmarks.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.writer = None

    def open(self):
        self.file = open(self.output_path, mode="w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "timestamp", "frame", "landmark", "raw_x", "raw_y", "raw_z",
            "filtered_x", "filtered_y", "filtered_z", "visibility", "presence",
            "posture", "fall_state", "torso_angle",
        ])

    def log(
        self,
        frame_number,
        landmarks,
        filtered_landmarks=None,
        posture=None,
        fall_state=None,
        torso_angle=None,
    ):
        """Log one row per landmark; optional arguments preserve old call compatibility."""
        if self.writer is None:
            raise RuntimeError("Logger is not open. Call open() first.")
        timestamp = datetime.now().isoformat()
        filtered_landmarks = filtered_landmarks or landmarks
        for index, raw in enumerate(landmarks):
            filtered = filtered_landmarks[index] if index < len(filtered_landmarks) else raw
            self.writer.writerow([
                timestamp,
                frame_number,
                index,
                raw.x,
                raw.y,
                getattr(raw, "z", None),
                filtered.x,
                filtered.y,
                getattr(filtered, "z", None),
                getattr(raw, "visibility", None),
                getattr(raw, "presence", None),
                getattr(posture, "value", posture),
                getattr(fall_state, "value", fall_state),
                torso_angle,
            ])
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None
