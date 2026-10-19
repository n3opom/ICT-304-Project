import cv2
import os


class Camera:
    """
    Camera class that supports two input modes:

    1. LIVE  - reads frames from the webcam
    2. IMAGE - reads pre-taken images from the test folder

    Both modes return an OpenCV frame.
    """

    def __init__(self, mode="live", image_folder=None, camera_index=0):
        if mode.lower() not in ("live", "image"):
            raise ValueError("Mode must be 'live' or 'image'")

        self.mode = mode.lower()
        self.image_folder = image_folder
        self.camera_index = camera_index
        self.cap = None

    def open(self):
        """
        Opens the selected input source.
        """

        if self.mode == "live":

            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam.")

        elif self.mode == "image":

            if self.image_folder is None:
                raise ValueError(
                    "An image folder must be provided when using image mode."
                )

            if not os.path.exists(self.image_folder):
                raise FileNotFoundError(
                    f"Image folder not found: {self.image_folder}"
                )

            self.image_files = []

            for filename in os.listdir(self.image_folder):

                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    self.image_files.append(
                        os.path.join(
                            self.image_folder,
                            filename
                        )
                    )

            self.image_files.sort()

            if len(self.image_files) == 0:
                raise FileNotFoundError(
                    f"No images found in: {self.image_folder}"
                )

        else:

            raise ValueError(
                "Invalid mode. Use 'live' or 'image'."
            )

    def read(self):
        """
        Gets the next frame.

        Returns:
            OpenCV frame, or None when no more frames are available.
        """

        if self.mode == "live":

            if self.cap is None:
                raise RuntimeError(
                    "Camera has not been opened. Call open() first."
                )

            ret, frame = self.cap.read()

            if not ret:
                return None

            return frame

        elif self.mode == "image":

            if self.image_index >= len(self.image_files):
                return None

            image_path = self.image_files[self.image_index]

            frame = cv2.imread(image_path)

            self.image_index += 1

            if frame is None:
                raise RuntimeError(
                    f"Could not read image: {image_path}"
                )

            return frame

    def release(self):
        """
        Releases the webcam.
        """

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_live(self):
        """
        Returns True if the camera is operating in live mode.
        """

        return self.mode == "live"


def show_frame(frame, window_name="Camera"):
    """
    Displays a frame in an OpenCV window.
    """

    if frame is None:
        return

    cv2.imshow(window_name, frame)


if __name__ == "__main__":

    print("Camera Test")
    print("-----------")
    print("1. Live webcam")
    print("2. Test images")

    choice = input("Select mode (1/2): ")

    # --------------------------------------------------
    # LIVE WEBCAM
    # --------------------------------------------------

    if choice == "1":

        camera = Camera(mode="live")

        try:

            camera.open()

            print()
            print("Live camera started.")
            print("Press Q to quit.")

            while True:

                frame = camera.read()

                if frame is None:
                    print("Could not read frame.")
                    break

                show_frame(frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

        finally:

            camera.release()
            cv2.destroyAllWindows()

    # --------------------------------------------------
    # TEST IMAGES
    # --------------------------------------------------

    elif choice == "2":

        image_folder = os.path.join(
            "data",
            "test"
        )

        camera = Camera(
            mode="image",
            image_folder=image_folder
        )

        try:

            camera.open()

            print()
            print("Test images loaded.")
            print("Press any key for the next image.")
            print("Press Q to quit.")

            while True:

                frame = camera.read()

                if frame is None:
                    print("All test images have been displayed.")
                    break

                show_frame(
                    frame,
                    "Test Image"
                )

                key = cv2.waitKey(0) & 0xFF

                if key == ord("q"):
                    break

        finally:

            camera.release()
            cv2.destroyAllWindows()

    else:

        print("Invalid choice.")