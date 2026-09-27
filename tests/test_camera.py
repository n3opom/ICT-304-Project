import os
import cv2

from src.camera import Camera


TEST_FOLDER = os.path.join(
    "data",
    "test"
)


def test_image_folder_exists():

    assert os.path.exists(
        TEST_FOLDER
    ), "Test image folder does not exist."


def test_images_exist():

    image_files = [
        filename
        for filename in os.listdir(TEST_FOLDER)
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    assert len(image_files) > 0, (
        "No test images found."
    )


def test_image_camera():

    camera = Camera(
        mode="image",
        image_folder=TEST_FOLDER
    )

    camera.open()

    frame = camera.read()

    assert frame is not None
    assert isinstance(frame, type(cv2.imread(
        os.path.join(
            TEST_FOLDER,
            os.listdir(TEST_FOLDER)[0]
        )
    )))

    camera.release()


def test_all_images_can_be_read():

    camera = Camera(
        mode="image",
        image_folder=TEST_FOLDER
    )

    camera.open()

    image_count = 0

    while True:

        frame = camera.read()

        if frame is None:
            break

        image_count += 1

    camera.release()

    assert image_count > 0


def test_image_only_returns_after_all_images():

    camera = Camera(
        mode="image",
        image_folder=TEST_FOLDER
    )

    camera.open()

    image_count = 0

    while True:

        frame = camera.read()

        if frame is None:
            break

        image_count += 1

    assert camera.read() is None

    camera.release()

    assert image_count > 0