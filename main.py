import cv2

from src.camera import Camera
from src.mediapipe import MediaPipePoseDetector
from src.landmark_logger import LandmarkLogger


# =========================
# Configuration
# =========================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30


def draw_landmarks(frame, pose_landmarks):
    """
    Draw MediaPipe Pose landmarks and connections
    onto an OpenCV frame.
    """

    height, width = frame.shape[:2]

    # MediaPipe Pose landmark connections
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

    # Create camera
    camera = Camera(
        mode="live",
        camera_index=CAMERA_INDEX
    )

    # Open webcam
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

    # Create MediaPipe detector
    detector = MediaPipePoseDetector()

    # Create landmark logger
    logger = LandmarkLogger(
        "results/landmarks.csv"
    )

    logger.open()

    print("======================================")
    print("ICT-304 MediaPipe Integration")
    print("======================================")
    print(f"Requested resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Requested FPS: {FRAME_FPS}")
    print("Press Q to quit.")
    print()

    frame_number = 0

    try:

        while True:

            # Read webcam frame
            frame = camera.read()

            if frame is None:
                print("Failed to read frame from camera.")
                break

            # Run MediaPipe Pose
            result = detector.detect(frame)

            # Increase frame counter
            frame_number += 1

            # Check whether a person was detected
            if result.pose_landmarks:

                # Use the first detected person
                pose_landmarks = result.pose_landmarks[0]

                # Save landmarks to CSV
                logger.log(
                    frame_number,
                    pose_landmarks
                )

                # Draw landmarks
                draw_landmarks(
                    frame,
                    pose_landmarks
                )

                # Display detection status
                cv2.putText(
                    frame,
                    "PERSON DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "NO PERSON DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # Display live video
            cv2.imshow(
                "ICT-304 MediaPipe Pose",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        logger.close()
        detector.close()
        camera.release()
        cv2.destroyAllWindows()

        print("Camera, MediaPipe and logger released.")


if __name__ == "__main__":
    main()