import os

import numpy as np
import yaml

from tiago_door_planning.planning.grips import grasp_action, grasp_rotation
from tiago_door_planning.ros.config import collision_params, goal_params, planner_params, scan_params

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
