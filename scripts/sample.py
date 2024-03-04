import pathlib

import average_face

average_face.compute_average_face(
    sorted(pathlib.Path("../data/samples").glob("*.jpg"), key=lambda x: int(x.stem.split("-")[2])),
    400,
    400,
    pathlib.Path("average-face-data"),
    "samples",
    auto_plot=True,
    plot_facet_col_wrap=5,
)
