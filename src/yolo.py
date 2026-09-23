from ultralytics import YOLO


class YOLODetector:
    """YOLO bounding-box detector."""

    def __init__(self, model_path="yolo11n.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image):
        """
        Detect objects in an image.

        Returns:
            List of detected objects containing:
            class name,
            confidence,
            and bounding box coordinates.
        """

        if image is None:
            raise ValueError("Image cannot be None.")

        results = self.model(
            image,
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

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    )
                })

        return detections

    def detect_persons(self, image):
        """
        Detect only people in an image.
        """

        detections = self.detect(image)

        persons = []

        for detection in detections:

            if detection["class_name"] == "person":
                persons.append(detection)

        return persons