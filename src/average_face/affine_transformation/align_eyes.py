from typing import TypeAlias, TypeVar, cast

import cv2
import numpy as np
import tqdm
from numpy.typing import NDArray

from .. import facial_landmarks
from ..typing import Image, Landmarks

__all__ = [
    "get_average_face_by_applying_affine_transformation_to_align_eyes",
    "apply_affine_transformation_to_align_eyes",
]

T = TypeVar("T", np.uint8, np.float64)


def get_average_face_by_applying_affine_transformation_to_align_eyes(
    images: list[Image],
    height: int,
    width: int,
) -> tuple[Image, list[Image], list[Landmarks], list[Landmarks]]:
    average_face_image = np.zeros((height, width, 3), np.float32)
    images_with_eyes_aligned = []
    landmarks_list = []
    landmarks_list_with_eyes_aligned = []
    for i, image in enumerate(tqdm.tqdm(images, desc="Aligning eyes", unit="image")):
        (
            im_with_eyes_aligned,
            landmarks_with_eye_aligned,
            landmarks,
        ) = apply_affine_transformation_to_align_eyes(
            image,
            width=width,
            height=height,
        )
        average_face_image[:] = average_face_image * (1 - 1 / (i + 1)) + im_with_eyes_aligned / (i + 1)
        images_with_eyes_aligned.append(im_with_eyes_aligned)
        landmarks_list.append(landmarks)
        landmarks_list_with_eyes_aligned.append(landmarks_with_eye_aligned)
    return (
        average_face_image.astype(np.uint8),
        images_with_eyes_aligned,
        landmarks_list,
        landmarks_list_with_eyes_aligned,
    )


def apply_affine_transformation_to_align_eyes(
    image: Image,
    *,
    width: int,
    height: int,
) -> tuple[Image, Landmarks, Landmarks]:
    landmarks = facial_landmarks.get_landmarks(image)

    # Outer corners of the eye in input image
    # Landmark index: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
    eye_corner_src = np.array(
        [
            landmarks[36],
            landmarks[45],
        ]
    )
    eye_corner_dst = np.array(
        [
            (int(0.3 * width), int(height / 3)),
            (int(0.7 * width), int(height / 3)),
        ]
    )

    mat = _compute_similarity_transform(eye_corner_src, eye_corner_dst)

    # Apply similarity transformation
    transformed_image = cv2.warpAffine(
        image,
        mat,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )
    transformed_landmarks = cv2.transform(
        landmarks[:, np.newaxis, :],
        mat,
    )[:, 0, :]

    return transformed_image, transformed_landmarks, landmarks


def _compute_similarity_transform(points_src: Landmarks, points_dst: Landmarks) -> NDArray[np.float32]:
    assert len(points_src) == 2
    assert len(points_dst) == 2
    mat, inliers = cv2.estimateAffinePartial2D(
        np.array([*points_src, _calculate_third_point_of_equilateral_triangle_from_two_points(*points_src)]),
        np.array([*points_dst, _calculate_third_point_of_equilateral_triangle_from_two_points(*points_dst)]),
    )
    return cast(NDArray[np.float32], mat)


def _calculate_third_point_of_equilateral_triangle_from_two_points(
    p1: NDArray[T], p2: NDArray[T]
) -> NDArray[np.float64]:
    return p1 + _rot60(p2 - p1)


def _rot60(p: NDArray[T]) -> NDArray[np.float64]:
    assert p.ndim == 1
    s60 = np.sin(60 * np.pi / 180)
    c60 = np.cos(60 * np.pi / 180)
    return cast(
        NDArray[np.float64],
        (
            np.array(
                [
                    [c60, -s60],
                    [s60, c60],
                ]
            )
            @ p
        ).astype(np.float64),
    )
