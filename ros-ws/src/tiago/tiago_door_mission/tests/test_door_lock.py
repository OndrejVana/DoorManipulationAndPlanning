import numpy as np

from tiago_door_mission.door_lock import DoorSample, LockParams, lock_door, mean_rotation

PARAMS = LockParams(min_samples=5, max_position_spread=0.02, max_yaw_spread=np.radians(3.0))


def pose(x, y, yaw):
    """Handle-like pose: z axis (the door normal) horizontal at yaw."""
    normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    result = np.eye(4)
    result[:3, :3] = np.column_stack([np.cross(up, normal), up, normal])
    result[:3, 3] = [x, y, 1.0]
    return result


def samples(count, noise=0.0, yaw_noise=0.0, tag_id=1, seed=0):
    rng = np.random.RandomState(seed)
    return [
        DoorSample(
            pose(*(rng.randn(2) * noise + [2.0, 0.0]), 0.0),
            pose(*(rng.randn(2) * noise + [2.0, 0.8]), np.pi + rng.randn() * yaw_noise),
            "left",
            True,
            tag_id,
        )
        for _ in range(count)
    ]


def test_agreeing_samples_lock_to_their_mean():
    door, reason = lock_door(samples(10, noise=0.005), PARAMS, "map")

    assert reason is None
    assert np.allclose(door.handle[:2, 3], [2.0, 0.8], atol=0.01)
    assert door.hinge_side == "left" and door.push and door.frame == "map"


def test_too_few_samples_do_not_lock():
    door, reason = lock_door(samples(3), PARAMS, "map")

    assert door is None
    assert "only 3" in reason


def test_scattered_samples_do_not_lock():
    door, reason = lock_door(samples(10, noise=0.05), PARAMS, "map")

    assert door is None
    assert "spread" in reason


def test_a_turning_normal_does_not_lock():
    door, reason = lock_door(samples(10, yaw_noise=np.radians(10.0)), PARAMS, "map")

    assert door is None
    assert "normal" in reason


def test_only_the_last_seen_tag_counts():
    mixed = samples(10, tag_id=2) + samples(6, tag_id=1)

    door, _ = lock_door(mixed, PARAMS, "map")

    assert door.tag_id == 1


def test_no_samples_mean_the_tag_was_not_seen():
    door, reason = lock_door([], PARAMS, "map")

    assert door is None
    assert "not seen" in reason


def test_mean_rotation_is_a_rotation():
    rotations = [pose(0.0, 0.0, yaw)[:3, :3] for yaw in (0.1, 0.2, 0.3)]

    rotation = mean_rotation(rotations)

    assert np.allclose(rotation @ rotation.T, np.eye(3))
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_a_few_outliers_are_left_out():
    good = samples(10, noise=0.003)
    jumped = [sample._replace(handle=sample.handle.copy()) for sample in good[:2]]
    for sample in jumped:
        sample.handle[:2, 3] += [0.5, 0.0]

    door, reason = lock_door(jumped + good, PARAMS, "map")

    assert reason is None
    assert np.allclose(door.handle[:2, 3], [2.0, 0.8], atol=0.01)


def test_of_two_groups_the_larger_one_locks():
    flipped = samples(8, noise=0.003, seed=1)
    for sample in flipped:
        sample.hinge[:2, 3] += [0.0, 1.5]
    door, reason = lock_door(flipped + samples(12, noise=0.003), PARAMS, "map")

    assert reason is None
    assert np.allclose(door.hinge[:2, 3], [2.0, 0.0], atol=0.01)
