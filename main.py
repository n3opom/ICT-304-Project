import cv2

from time import monotonic

from src.camera import Camera
from src.mediapipe import MediaPipePoseDetector
from src.temporal_fall_detector import (
    TemporalFallDetector,
    FallState,
    Posture,
)
from src.landmark_logger import LandmarkLogger


# =========================
# Configuration
# =========================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30

POSSIBLE_FALL_ANGLE = 60.0
MIN_LANDMARK_VISIBILITY = 0.5

RAPID_TRANSITION_SECONDS = 1.0
LOW_POSTURE_SECONDS = 1.0
RAPID_MOVEMENT_THRESHOLD = 0.08


def draw_landmarks(frame, pose_landmarks):
    """
    Draw MediaPipe Pose landmarks and connections
    onto an OpenCV frame.
    """

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

    # Draw connections
    for start_index, end_index in connections:
        start = pose_landmarks[start_index]
        end = pose_landmarks[end_index]

        start_point = (
            int(start.x * width),
            int(start.y * height)
        )

        end_point = (
            int(end.x * width),
            int(end.y * height)
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            (0, 255, 0),
            2
        )

    # Draw landmarks
    for landmark in pose_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


def main():

    camera = None
    detector = None
    logger = None

    try:
        # =========================
        # Initialize components
        # =========================

        camera = Camera(
            mode="live",
            camera_index=CAMERA_INDEX
        )

        camera.open()

        # Configure webcam
        camera.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        camera.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        camera.cap.set(
            cv2.CAP_PROP_FPS,
            FRAME_FPS
        )

        # Create MediaPipe pose detector
        detector = MediaPipePoseDetector()

        # Create temporal fall detector
        fall_detector = TemporalFallDetector(
            min_landmark_visibility=MIN_LANDMARK_VISIBILITY,
            rapid_transition_seconds=RAPID_TRANSITION_SECONDS,
            low_posture_seconds=LOW_POSTURE_SECONDS,
            possible_fall_angle=POSSIBLE_FALL_ANGLE,
            rapid_movement_threshold=RAPID_MOVEMENT_THRESHOLD,
        )

        # Create landmark logger
        logger = LandmarkLogger(
            "results/landmarks.csv"
        )

        logger.open()

        print("======================================")
        print("ICT-304 MediaPipe Fall Detection")
        print("======================================")
        print(
            f"Requested resolution: "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT}"
        )
        print(f"Requested FPS: {FRAME_FPS}")
        print(
            f"Possible fall angle threshold: "
            f"{POSSIBLE_FALL_ANGLE} degrees"
        )
        print("Press Q to quit.")
        print()

        frame_number = 0

        # =========================
        # Main camera loop
        # =========================

        while True:

            frame = camera.read()

            if frame is None:
                print("Failed to read frame from camera.")
                break

            # Detect pose
            result = detector.detect(frame)

            frame_number += 1


            if result.pose_landmarks:
                # Use the first detected person.
                pose_landmarks = result.pose_landmarks[0]

                # Update the temporal detector using this frame's timestamp.
                fall_state, posture = fall_detector.update(
                    pose_landmarks,
                    timestamp=monotonic(),
                )

                # Save landmarks to CSV.
                logger.log(frame_number, pose_landmarks)

                # Draw pose landmarks.
                draw_landmarks(frame, pose_landmarks)

                # Display posture.
                posture_text = f"Posture: {posture.value}"
                cv2.putText(
                    frame,
                    posture_text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Display temporal fall state.
                if fall_state == FallState.FALL:
                    display_text = "FALL"
                    display_color = (0, 0, 255)
                elif fall_state == FallState.POSSIBLE_FALL:
                    display_text = "POSSIBLE FALL"
                    display_color = (0, 165, 255)
                else:
                    display_text = "NO FALL"
                    display_color = (0, 255, 0)

                cv2.putText(
                    frame,
                    display_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    display_color,
                    2,
                )

                # Display torso angle for debugging.
                features = fall_detector._features(pose_landmarks)
                torso_angle = features[0] if features is not None else None

                angle_text = (
                    f"Torso angle: {torso_angle:.1f} deg"
                    if torso_angle is not None
                    else "Torso angle: unavailable"
                )

                cv2.putText(
                    frame,
                    angle_text,
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            else:
                # Let the temporal detector know no pose was detected.
                fall_state, posture = fall_detector.update(
                    None,
                    timestamp=monotonic(),
                )

                cv2.putText(
                    frame,
                    "NO PERSON DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            # Display live video
            cv2.imshow(
                "ICT-304 MediaPipe Fall Detection",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # =========================
        # Release resources
        # =========================

        if logger is not None:
            logger.close()

        if detector is not None:
            detector.close()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print("Camera, MediaPipe and logger released.")


if __name__ == "__main__":
    main()
