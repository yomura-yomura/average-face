import logging
import pathlib

import cv2
import numpy as np

from . import affine_transformation, io, plotly
from .typing import Image

logger = logging.getLogger(__name__)


def compute_average_face(
    image_paths: list[pathlib.Path],
    height: int,
    width: int,
    dir_to_save: pathlib.Path | None = None,
    save_dirname: str | None = None,
    auto_plot: bool = True,
    plot_facet_col_wrap: int = 9,
) -> tuple[Image, Image]:
    logging.info("Step 0: Load images")

    images = [io.load_image(image_path) for image_path in image_paths]

    logger.info("Step 1: Align eyes")

    (
        average_face_image,
        images_with_eyes_aligned,
        landmarks_list,
        landmarks_list_with_eyes_aligned,
    ) = affine_transformation.get_average_face_by_applying_affine_transformation_to_align_eyes(
        images,
        height,
        width,
    )

    if auto_plot:
        logger.info("Plotting images with eyes aligned")
        plotly.plot_faces(
            images_with_eyes_aligned,
            landmarks_list_with_eyes_aligned,
            facet_col_wrap=plot_facet_col_wrap,
            title="Similarity Transformed Images with Eyes Aligned",
        )

    if dir_to_save is not None:
        assert save_dirname is not None, "save_dirname must be provided if dir_to_save is provided"
        logger.info(f"Saving images with eyes aligned at {dir_to_save / 'aligned-eyes' / save_dirname}")
        for i, image in enumerate(images_with_eyes_aligned):
            filepath_to_save = (dir_to_save / "aligned-eyes" / save_dirname / f"{i}").with_suffix(".jpg")
            filepath_to_save.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(
                str(filepath_to_save),
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )

        logging.info(f"Saving average face at {dir_to_save / 'average-face' / save_dirname}")
        filepath_to_save = (dir_to_save / "average-face" / save_dirname).with_suffix(".jpg")
        filepath_to_save.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(
            str(filepath_to_save),
            cv2.cvtColor(average_face_image, cv2.COLOR_BGR2RGB),
        )

    logger.info("Step 2: Warp triangles / Step 3: Average")

    (
        average_face_image2,
        images_with_warped_triangles,
    ) = affine_transformation.get_average_face_by_applying_affine_transformation_for_each_triangle(
        images,
        landmarks_list,
        np.mean(landmarks_list_with_eyes_aligned, axis=0),
        width,
        height,
    )
    # Fill in black pixels with the v1 average face image
    average_face_image2[average_face_image2 == 0] = average_face_image[average_face_image2 == 0]

    if auto_plot:
        logging.info("Plotting images with warped triangles")
        plotly.plot_faces(
            images_with_warped_triangles,
            [np.mean(landmarks_list_with_eyes_aligned, axis=0)] * len(images_with_warped_triangles),
            facet_col_wrap=plot_facet_col_wrap,
            title="Similarity Transformed Images with Eyes Aligned and Warped Triangles",
        )

        logging.info("Plotting average face")
        plotly.plot_average_face(
            average_face_image,
            average_face_image2,
        )

    if dir_to_save is not None:
        assert save_dirname is not None, "save_dirname must be provided if dir_to_save is provided"
        logging.info(f"Saving images with warped triangles at {dir_to_save / 'warped-triangles' / save_dirname}")
        for i, image in enumerate(images_with_warped_triangles):
            filepath_to_save = (dir_to_save / "warped-triangles" / save_dirname / f"{i}").with_suffix(".jpg")
            filepath_to_save.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(
                str(filepath_to_save),
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )

        logging.info(f"Saving average face at {dir_to_save / 'average-face2' / save_dirname}")
        filepath_to_save = (dir_to_save / "average-face2" / save_dirname).with_suffix(".jpg")
        filepath_to_save.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(
            str(filepath_to_save),
            cv2.cvtColor(average_face_image2, cv2.COLOR_BGR2RGB),
        )

    return average_face_image, average_face_image2
