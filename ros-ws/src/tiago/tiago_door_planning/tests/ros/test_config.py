import os

import numpy as np
import pytest
import yaml

from tiago_door_planning.planning.grips import grasp_action, grasp_rotation
from tiago_door_planning.ros.config import (
    apply_pipeline_speed,
    collision_params,
    goal_params,
    planner_params,
    scan_params,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "door_motion.yaml")


def load_config():
    with open(CONFIG_PATH) as config_file:
        return yaml.safe_load(config_file)


def test_shipped_config_builds_all_parameter_sets():
    config = load_config()

    assert planner_params(config).angle_step == np.radians(config["planner"]["angle_step_deg"])
    assert goal_params(config).preferred_angle <= np.pi / 2.0 + 1e-9
    assert collision_params(config).base_radius > 0.0
    assert scan_params(config).max_angle > 0.0
    for action in ("pull", "push"):
        assert np.allclose(grasp_rotation(config, action) @ grasp_rotation(config, action).T, np.eye(3))


def test_pull_door_back_handle_is_a_push_grasp():
    assert grasp_action(False, "front") == "pull"
    assert grasp_action(False, "back") == "push"
    assert grasp_action(True, "front") == "push"


def speed_config(level):
    return {
        "speed": {
            "pipeline_speed": level,
            "fast": {"planner": {"linear_speed": 0.6}, "grasp": {"hand_duration_s": 1.0}},
            "durations": ["hand_duration_s"],
        },
        "planner": {"linear_speed": 0.2, "angular_speed": 0.5},
        "grasp": {"hand_duration_s": 2.0},
    }


def scaled(level):
    config = apply_pipeline_speed(speed_config(level))
    return config["planner"]["linear_speed"], config["grasp"]["hand_duration_s"]


def test_pipeline_speed_runs_from_half_through_today_to_fast():
    assert np.allclose(scaled(0.0), [0.1, 4.0])
    assert np.allclose(scaled(0.5), [0.2, 2.0])
    assert np.allclose(scaled(1.0), [0.6, 1.0])


def test_pipeline_speed_interpolates_rates_and_durations_alike():
    assert np.allclose(scaled(0.25), [0.15, 1.0 / 0.375])
    assert np.allclose(scaled(0.75), [0.4, 1.0 / 0.75])


def test_a_key_kept_from_going_below_today_still_speeds_up():
    config = speed_config(0.0)
    config["speed"]["not_below_today"] = ["linear_speed"]
    assert np.isclose(apply_pipeline_speed(config)["planner"]["linear_speed"], 0.2)

    config["speed"]["pipeline_speed"] = 1.0
    assert np.isclose(apply_pipeline_speed(config)["planner"]["linear_speed"], 0.6)


def test_pipeline_speed_outside_zero_to_one_is_clipped():
    assert np.allclose(scaled(-1.0), scaled(0.0))
    assert np.allclose(scaled(2.0), scaled(1.0))


def test_pipeline_speed_leaves_unlisted_keys_and_the_input_alone():
    config = speed_config(1.0)
    apply_pipeline_speed(config)

    assert config["planner"]["linear_speed"] == 0.2
    assert apply_pipeline_speed(config)["planner"]["angular_speed"] == 0.5


def test_shipped_pipeline_speed_keeps_the_tested_values():
    config = load_config()

    scaled_config = apply_pipeline_speed(config)

    assert config["speed"]["pipeline_speed"] == 0.5
    for section, fast_values in config["speed"]["fast"].items():
        for name in fast_values:
            assert np.isclose(scaled_config[section][name], config[section][name])


def test_pushing_plans_with_its_own_door_rate():
    config = load_config()
    config["planner"].update(door_rate_deg=15.0, push_door_rate_deg=10.0)

    assert planner_params(config).door_rate == pytest.approx(np.radians(15.0))
    assert planner_params(config, push=True).door_rate == pytest.approx(np.radians(10.0))
