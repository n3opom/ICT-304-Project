from src.yolo_detection import YOLODetector


def test_yolo_model_loads():
    detector = YOLODetector()

    assert detector.model is not None


def test_yolo_none_frame():
    detector = YOLODetector()

    try:
        detector.detect(None)
        assert False
    except ValueError:
        assert True


def test_yolo_confidence_value():
    detector = YOLODetector(confidence=0.6)

    assert detector.confidence == 0.6