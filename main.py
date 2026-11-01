
import cv2

from time import monotonic

from src.camera import Camera
from src.mediapipe import MediaPipePoseDetector
from src.landmark_logger import LandmarkLogger
from src.ml_fall_detector import MLFallDetector

from train_fall_classifier import frame_features


# ==================================================
# Configuration
# ==================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30


# ==================================================
# Draw MediaPipe landmarks
# ==================================================

def draw_landmarks(frame, pose_landmarks):

    height, width = frame.shape[:2]

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12),
        (11, 13), (13, 15),
        (15, 17), (15, 19), (15, 21),
        (17, 19),
        (12, 14), (14, 16),
        (16, 18), (16, 20), (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (25, 27),
        (27, 29),
        (29, 31),
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32),
        (27, 28),
    ]

    for start_index, end_index in connections:

        start = pose_landmarks[start_index]
        end = pose_landmarks[end_index]

        start_point = (
            int(start.x * width),
            int(start.y * height),
        )

        end_point = (
            int(end.x * width),
            int(end.y * height),
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    for landmark in pose_landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1,
        )


# ==================================================
# Display helper
# ==================================================

def display_text(
    frame,
    text,
    y,
    color=(255, 255, 255),
    scale=0.7,
):

    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )


# ==================================================
# Main application
# ==================================================

def main():

    camera = None
    pose_detector = None
    logger = None

    try:

        # ------------------------------------------
        # Initialize webcam
        # ------------------------------------------

        camera = Camera(
            mode="live",
            camera_index=CAMERA_INDEX,
        )

        camera.open()

        camera.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH,
        )

        camera.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT,
        )

        camera.cap.set(
            cv2.CAP_PROP_FPS,
            FRAME_FPS,
        )

        # ------------------------------------------
        # Initialize MediaPipe
        # ------------------------------------------

        pose_detector = MediaPipePoseDetector()

        # ------------------------------------------
        # Load Random Forest V3
        # ------------------------------------------

        fall_detector = MLFallDetector()

        # ------------------------------------------
        # Initialize landmark logger
        # ------------------------------------------

        logger = LandmarkLogger(
            "results/landmarks.csv"
        )

        logger.open()

        print("======================================")
        print("ICT-304 Random Forest V3 Fall Detection")
        print("======================================")

        print(
            f"Requested resolution: "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT}"
        )

        print(f"Requested FPS: {FRAME_FPS}")

        print(
            "Model: models/fall_classifier_v3.joblib"
        )

        print("Fall probability threshold: 50%")
        print("Required positive predictions: 2")
        print("Evidence window: 0.5 seconds")
        print("Alert cooldown: 8 seconds")

        print("\nPress Q to quit.\n")

        frame_number = 0

        # ------------------------------------------
        # Main webcam loop
        # ------------------------------------------

        while True:

            frame = camera.read()

            if frame is None:
                print("Failed to read frame.")
                break

            now = monotonic()

            # Detect body pose using existing MediaPipe.
            result = pose_detector.detect(frame)

            frame_number += 1

            pose_landmarks = (
                result.pose_landmarks[0]
                if result.pose_landmarks
                else None
            )

            # --------------------------------------
            # Random Forest V3 inference
            # --------------------------------------

            (
                fall_probability,
                predicted_fall,
                alert_triggered,
                alert_active,
            ) = fall_detector.update(
                pose_landmarks,
                timestamp=now,
            )

            # --------------------------------------
            # Person detected
            # --------------------------------------

            if pose_landmarks is not None:

                draw_landmarks(
                    frame,
                    pose_landmarks,
                )

                # Show the same torso-angle feature
                # used by the Random Forest model.
                features = frame_features(
                    pose_landmarks
                )

                torso_angle = (
                    float(features[0]) * 90.0
                    if features is not None
                    else None
                )

                # A probability threshold crossing is
                # not itself a confirmed fall.
                if alert_active:
                    fall_state_text = "POTENTIAL FALL ALERT"

                elif predicted_fall:
                    fall_state_text = "FALL-LIKE MOVEMENT"

                else:
                    fall_state_text = "NO FALL ALERT"

                logger.log(
                    frame_number,
                    pose_landmarks,
                    filtered_landmarks=pose_landmarks,
                    posture=None,
                    fall_state=fall_state_text,
                    torso_angle=torso_angle,
                )

                display_text(
                    frame,
                    "MediaPipe: PERSON DETECTED",
                    80,
                )

                if torso_angle is not None:
                    display_text(
                        frame,
                        f"Torso angle: {torso_angle:.1f} deg",
                        120,
                    )

            else:

                fall_state_text = "NO PERSON DETECTED"

            # --------------------------------------
            # Display fall probability
            # --------------------------------------

            if fall_probability is None:
                probability_text = (
                    "Fall probability: unavailable"
                )

            else:
                probability_text = (
                    f"Fall probability: "
                    f"{fall_probability * 100:.1f}%"
                )

            # --------------------------------------
            # Select alert display
            # --------------------------------------

            if alert_active:

                status_text = "POTENTIAL FALL ALERT"
                status_color = (0, 0, 255)

            elif predicted_fall:

                status_text = "FALL-LIKE MOVEMENT"
                status_color = (0, 165, 255)

            elif pose_landmarks is None:

                status_text = "NO PERSON DETECTED"
                status_color = (0, 0, 255)

            else:

                status_text = "NO FALL ALERT"
                status_color = (0, 255, 0)

            display_text(
                frame,
                status_text,
                40,
                status_color,
                1.0,
            )

            display_text(
                frame,
                probability_text,
                160,
            )

            display_text(
                frame,
                "Detector: Random Forest V3",
                200,
            )

            # --------------------------------------
            # Record potential fall event
            # --------------------------------------

            if alert_triggered:

                print(
                    f"[POTENTIAL FALL] "
                    f"Frame {frame_number}, "
                    f"probability: "
                    f"{fall_probability:.2%}"
                )

            # --------------------------------------
            # Show video
            # --------------------------------------

            cv2.imshow(
                "ICT-304 Random Forest V3 Fall Detection",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        if logger is not None:
            logger.close()

        if pose_detector is not None:
            pose_detector.close()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print(
            "Camera, MediaPipe, and logger released."
        )


if __name__ == "__main__":
    main()
