import logging

import numpy as np
import plotly.express as px
import plotly.graph_objs as go

from .typing import Image, Landmarks

__all__ = ["plot_faces", "plot_average_face"]

logger = logging.getLogger(__name__)


def plot_faces(
    images: list[Image],
    landmarks_list: list[Landmarks] | None = None,
    *,
    facet_col_wrap: int = 9,
    title: str | None = None,
    auto_plot: bool = True,
) -> go.Figure:
    image_heights = [image.shape[0] for image in images]
    image_widths = [image.shape[1] for image in images]
    logger.debug(f"Image heights: min={min(image_heights)}, max={max(image_heights)}, mean={np.mean(image_heights)}")
    logger.debug(f"Image widths: min={min(image_widths)}, max={max(image_widths)}, mean={np.mean(image_widths)}")

    fig = px.imshow(
        np.asarray(images),
        title=title,
        facet_col=0,
        facet_col_wrap=facet_col_wrap,
        facet_col_spacing=0.01,
        facet_row_spacing=0.01,
        height=100 + int(150 * (len(images) / facet_col_wrap + 1) * max(image_heights) / 300),
        width=200 + int(150 * facet_col_wrap * max(image_widths) / 300),
    )
    fig.update_annotations(text="")
    if landmarks_list is not None and len(landmarks_list) > 0:
        estimated_facet_rows = int(np.ceil(len(landmarks_list) / facet_col_wrap))
        logger.debug(f"{estimated_facet_rows = }")
        for i, landmarks in enumerate(landmarks_list):
            fig.add_trace(
                go.Scattergl(
                    x=landmarks[:, 0],
                    y=landmarks[:, 1],
                    mode="markers",
                    marker=dict(
                        color="red",
                        size=3,
                    ),
                    legendgroup="landmarks",
                    name="Landmarks",
                    showlegend=False,
                ),
                row=estimated_facet_rows - i // facet_col_wrap,
                col=i % facet_col_wrap + 1,
            )
        fig.data[-1].update(showlegend=True)

    if auto_plot:
        fig.show()
    return fig


def plot_average_face(
    *average_face_images: Image,
    auto_plot: bool = True,
) -> go.Figure:
    fig = px.imshow(
        np.array(average_face_images),
        title="Average Face",
        facet_col=0,
        facet_col_wrap=2,
        labels={"facet_col": "version"},
    )
    if auto_plot:
        fig.show()
    return fig
