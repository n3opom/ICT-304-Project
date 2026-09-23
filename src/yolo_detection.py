from ultralytics import YOLO


class YOLODetector:
    """YOLO object detection wrapper."""

    def __init__(self, model_path="yolo11n.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Run YOLO detection on an OpenCV image.

        Args:
            frame: OpenCV image/frame.

        Returns:
            List of detection dictionaries.
        """

        if frame is None:
            raise ValueError("Frame cannot be None.")

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = self.model.names[class_id]

                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    )
                }

                detections.append(detection)

        return detections