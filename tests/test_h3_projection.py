# -*- coding: utf-8 -*-

from spatial_encoder.v26.h3_projection import (
    line_to_cells,
    point_to_cell,
    polygon_to_cells,
)


def test_point_to_cell_returns_single_h3_cell():
    cell = point_to_cell(114.364, 30.532, 9)

    assert isinstance(cell, str)
    assert cell


def test_line_to_cells_returns_multi_cell_coverage():
    coords = [(114.360, 30.530), (114.365, 30.535)]
    cells = line_to_cells(coords, 9)

    assert len(cells) >= 1
    assert all(isinstance(cell, str) for cell in cells)


def test_polygon_to_cells_returns_weighted_coverage():
    polygon = [
        (114.360, 30.530),
        (114.366, 30.530),
        (114.366, 30.536),
        (114.360, 30.536),
        (114.360, 30.530),
    ]
    cells = polygon_to_cells(polygon, 9)

    assert len(cells) >= 1
    assert all("cell" in item and "weight" in item for item in cells)
