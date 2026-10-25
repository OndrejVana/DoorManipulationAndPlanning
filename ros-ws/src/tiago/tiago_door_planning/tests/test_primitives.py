import numpy as np
import pytest

from tiago_door_planning.geometry import wrap_angle
from tiago_door_planning.primitives import heading_angle, make_primitive, primitive_table

RESOLUTION = 0.05
BINS = 16


@pytest.mark.parametrize("heading_bin", range(BINS))
def test_every_primitive_ends_exactly_on_the_lattice(heading_bin):
    for primitive in primitive_table(BINS, RESOLUTION, 4, 6)[heading_bin]:
        end = primitive.samples[-1]
        expected_heading = heading_angle(heading_bin + primitive.heading_change, BINS)

        assert end[:2] == pytest.approx(np.array(primitive.cell_offset) * RESOLUTION)
        assert wrap_angle(end[2] - expected_heading) == pytest.approx(0.0, abs=1e-9)


def test_forward_curve_moves_along_its_heading():
    primitive = make_primitive(1, 0, 4, BINS, RESOLUTION, 20)
    steps = np.diff(primitive.samples[:, :2], axis=0)
    motion_headings = np.arctan2(steps[:, 1], steps[:, 0])

    assert np.max(np.abs(wrap_angle(motion_headings - primitive.samples[1:, 2]))) < np.radians(15.0)


def test_reverse_primitive_moves_backwards():
    primitive = make_primitive(0, 0, -4, BINS, RESOLUTION, 6)

    assert primitive.reverse
    assert primitive.cell_offset == (-4, 0)


def test_rotation_stays_in_place():
    primitive = make_primitive(3, 1, 0, BINS, RESOLUTION, 6)

    assert primitive.cell_offset == (0, 0)
    assert np.allclose(primitive.samples[:, :2], 0.0)
    assert primitive.turn == pytest.approx(2.0 * np.pi / BINS)
