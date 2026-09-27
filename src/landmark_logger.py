import csv
from pathlib import Path
from datetime import datetime


class LandmarkLogger:
    """
    Logs MediaPipe Pose landmarks to a CSV file.

    Each detected landmark is stored as one row.
    """

    def __init__(self, output_path="results/landmarks.csv"):
        self.output_path = Path(output_path)

        # Create the parent directory if it does not exist
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.file = None
        self.writer = None

    def open(self):
        """Open the CSV file and write the header."""

        self.file = open(
            self.output_path,
            mode="w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp",
            "frame",
            "landmark",
            "x",
            "y",
            "z",
            "visibility",
            "presence"
        ])

    def log(self, frame_number, landmarks):
        """
        Log all landmarks for one frame.

        Args:
            frame_number: Current webcam frame number.
            landmarks: MediaPipe pose landmark list.
        """

        if self.writer is None:
            raise RuntimeError(
                "Logger is not open. Call open() first."
            )

        timestamp = datetime.now().isoformat()

        for landmark_index, landmark in enumerate(landmarks):

            visibility = getattr(
                landmark,
                "visibility",
                None
            )

            presence = getattr(
                landmark,
                "presence",
                None
            )

            self.writer.writerow([
                timestamp,
                frame_number,
                landmark_index,
                landmark.x,
                landmark.y,
                landmark.z,
                visibility,
                presence
            ])

        # Make sure data is written to disk
        self.file.flush()

    def close(self):
        """Close the CSV file."""

        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None