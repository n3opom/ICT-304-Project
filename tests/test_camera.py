import cv2
import pytest

from src.camera import Camera


def test_camera_initialises_in_live_mode():
    camera = Camera(
        mode="live"
    )

    assert camera.mode == "live"
    assert camera.camera_index == 0

    camera.release()


def test_camera_rejects_invalid_mode():
    with pytest.raises(ValueError):
        Camera(
            mode="invalid"
        )


def test_camera_image_mode_requires_folder():
    camera = Camera(
        mode="image"
    )

    with pytest.raises(ValueError):
        camera.open()


def test_camera_live_mode_status():
    camera = Camera(
        mode="live"
    )

    assert camera.is_live() is True

    camera.release()


def test_camera_image_mode_status():
    camera = Camera(
        mode="image",
        image_folder="data/test"
    )

    assert camera.is_live() is False

    camera.release()