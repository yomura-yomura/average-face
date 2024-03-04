import logging

import cv2
import numpy as np
import tqdm
from numpy.typing import NDArray
from scipy.spatial import Delaunay

from ..typing import Image, Landmarks

logger = logging.getLogger(__name__)


__all__ = [
    "get_average_face_by_applying_affine_transformation_for_each_triangle",
    "apply_affine_transformation_for_each_triangle",
]


def get_average_face_by_applying_affine_transformation_for_each_triangle(
    images: list[Image],
    landmarks_list: list[Landmarks],
    averaged_landmarks_with_eyes_aligned: Landmarks,
    width: int,
    height: int,
) -> tuple[Image, list[Image]]:
    delaunay = Delaunay(averaged_landmarks_with_eyes_aligned)
    logger.debug(f"{delaunay.simplices.shape = }")

    transformed_images = []
    average_face_image = np.zeros((height, width, 3), np.float32)
    for i, (image, landmarks) in enumerate(
        zip(
            tqdm.tqdm(images, desc="Warping triangles", unit="image"),
            landmarks_list,
            strict=True,
        )
    ):
        transformed_image = apply_affine_transformation_for_each_triangle(
            image,
            landmarks,
            averaged_landmarks_with_eyes_aligned,
            delaunay.simplices,
            width,
            height,
        )
        transformed_images.append(transformed_image)
        average_face_image[:] = average_face_image * (1 - 1 / (i + 1)) + transformed_image / (i + 1)
    return (
        average_face_image.astype(np.uint8),
        transformed_images,
    )


def apply_affine_transformation_for_each_triangle(
    image: Image,
    landmarks: Landmarks,
    averaged_landmarks_with_eyes_aligned: Landmarks,
    delaunay_simplices: NDArray[np.int64],
    width: int,
    height: int,
) -> Image:
    transformed_image = np.zeros((height, width, 3), np.uint8)

    # Apply affine transformation for each triangle to warp triangles
    for j, (landmarks_in_triangle, averaged_landmarks_with_eyes_aligned_in_triangle) in enumerate(
        zip(
            landmarks[delaunay_simplices],
            averaged_landmarks_with_eyes_aligned[delaunay_simplices],
            strict=True,
        )
    ):
        _warp_triangle(
            image,
            transformed_image,
            landmarks_in_triangle,
            averaged_landmarks_with_eyes_aligned_in_triangle,
            # alpha_to_blend_in_outer_triangle=1 / len(delaunay.simplices),
        )
    return transformed_image


def _warp_triangle(
    img_src: Image,
    img_dst: Image,
    points_src: Landmarks,
    points_dst: Landmarks,
    alpha_to_blend_in_outer_triangle: float = 0.0,
) -> None:
    # Find bounding rectangle for each triangle: bb = (x, y, w, h)
    bb_src = cv2.boundingRect(np.asarray([points_src], dtype=np.float32))
    bb_dst = cv2.boundingRect(np.asarray([points_dst], dtype=np.float32))
    logger.debug(f"{bb_src = }, {bb_dst = }")

    # Offset points by left top corner of the respective bounding rectangles
    points_rect_src = points_src - bb_src[:2]
    points_rect_dst = points_dst - bb_dst[:2]

    # Get mask by filling triangle
    mask_dst = np.zeros((bb_dst[3], bb_dst[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask_dst, points_rect_dst.astype(np.int32), (1.0, 1.0, 1.0), cv2.LINE_AA, 0)
    assert np.unique(mask_dst).tolist() == [0.0, 1.0], f"{np.unique(mask_dst) = }"
    mask_dst = mask_dst.astype(bool)

    # Apply the affine transformation to the rectangular patch to warp points_rect_src to points_rect_dst.
    transformed_img_dst_rect = cv2.warpAffine(
        img_src[bb_src[1] : bb_src[1] + bb_src[3], bb_src[0] : bb_src[0] + bb_src[2]],
        cv2.getAffineTransform(points_rect_src.astype("f4"), points_rect_dst.astype("f4")),
        (bb_dst[2], bb_dst[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Copy triangular inner region of the rectangular patch to the destination image.
    img_dst_rect = img_dst[
        bb_dst[1] : bb_dst[1] + bb_dst[3],
        bb_dst[0] : bb_dst[0] + bb_dst[2],
    ]
    img_dst_rect[mask_dst] = transformed_img_dst_rect[mask_dst]

    # Blend triangular outer region of the rectangular patch and the destination image with alpha.
    img_dst_rect[~mask_dst] = (
        img_dst_rect[~mask_dst] * (1 - alpha_to_blend_in_outer_triangle)
        + transformed_img_dst_rect[~mask_dst] * alpha_to_blend_in_outer_triangle
    )
