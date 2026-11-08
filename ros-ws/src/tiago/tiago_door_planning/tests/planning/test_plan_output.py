import numpy as np
import pytest

from tiago_door_planning.planning.door_planner import Grip, PathStep
from tiago_door_planning.planning.plan_output import plan_to_segments

START = PathStep([np.array([0.0, 0.0, 0.0])], [0.0], 0.0, True, np.zeros(2), Grip("left", "front"))


def move(x_end, angle_end, grasped, duration=1.0, grip=Grip("left", "front")):
    poses = [np.array([x_end - 0.05, 0.0, 0.0]), np.array([x_end, 0.0, 0.0])]
    return PathStep(poses, [angle_end, angle_end], duration, grasped, np.ones(2) * x_end, grip)


def regrasp(x, angle, grasped, grip=Grip("left", "front")):
    return PathStep([np.array([x, 0.0, 0.0])], [angle], 7.5, grasped, np.full(2, 9.0), grip)


def regrasp_plan():
    return [START, move(0.1, 0.1, True), regrasp(0.1, 0.1, False), move(0.2, 0.1, False), regrasp(0.2, 0.1, True),
            move(0.3, 0.2, True)]


def test_segments_skip_regrasp_and_resample_uniformly():
    steps = regrasp_plan()

    segments = plan_to_segments(steps)

    assert [segment.grasped for segment in segments] == [True, False, True]
    first = segments[0].trajectory
    assert first.base_poses[0] == pytest.approx([0.0, 0.0, 0.0])
    assert first.base_poses[-1] == pytest.approx([0.1, 0.0, 0.0])
    assert len(first.door_angles) == 11


def test_grasped_segment_after_regrasp_starts_from_grasp_configuration():
    steps = regrasp_plan()

    last = plan_to_segments(steps)[-1].trajectory

    assert last.arm_qs[0] == pytest.approx(np.full(2, 9.0))
    assert last.base_poses[0] == pytest.approx([0.2, 0.0, 0.0])


def test_segments_carry_the_arm_that_holds_or_let_go():
    steps = [
        START,
        move(0.1, 0.1, True),
        regrasp(0.1, 0.1, False),
        move(0.2, 0.1, False),
        regrasp(0.2, 0.1, True, Grip("right", "back")),
        move(0.3, 0.2, True, grip=Grip("right", "back")),
    ]

    segments = plan_to_segments(steps)

    assert [segment.trajectory.side for segment in segments] == ["left", "left", "right"]
    assert [segment.face for segment in segments] == ["front", "front", "back"]


def test_arm_configuration_follows_the_plan_between_steps():
    steps = [START, move(0.1, 0.1, True), move(0.2, 0.2, True)]

    arm_qs = plan_to_segments(steps)[0].trajectory.arm_qs

    assert arm_qs[0] == pytest.approx(np.zeros(2))
    assert arm_qs[1] == pytest.approx(np.full(2, 0.01))
    assert arm_qs[-1] == pytest.approx(np.full(2, 0.2))
