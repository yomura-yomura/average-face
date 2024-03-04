import cv2
import dlib
import numpy as np

from .typing import Image, Landmarks

__all__ = ["get_landmarks"]

# See https://github.com/davisking/dlib-models?tab=readme-ov-file#shape_predictor_68_face_landmarksdatbz2
SHAPE_PREDICTOR_68_FACE_LANDMARKS_DAT = "shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_68_FACE_LANDMARKS_DAT)

frontal_face_detector = dlib.get_frontal_face_detector()


def get_landmarks(image: Image) -> Landmarks:
    rects = frontal_face_detector(image, 0)
    assert len(rects) == 1

    landmarks = np.array(
        [
            (p.x, p.y)
            for p in shape_predictor(
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                rects[0],
            ).parts()
        ]
    )
    assert landmarks.shape == (68, 2)
    return landmarks
