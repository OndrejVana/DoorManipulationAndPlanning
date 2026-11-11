import copy

import numpy as np

from ..model.collision import CollisionParams
from ..planning.door_goal import GoalParams
from ..planning.door_planner import PlannerParams
from ..model.door_scene import DoorDimensions
from ..sensing.door_sensing import ScanParams


def section_params(params_type, section, **fixed):
    values = dict(fixed)
    for field in params_type._fields:
        if field in section:
            values.setdefault(field, section[field])
        elif field + "_deg" in section:
            values.setdefault(field, np.radians(section[field + "_deg"]))
    return params_type(**values)


def door_dimensions(config):
    return section_params(DoorDimensions, config["door"])


def collision_params(config):
    return section_params(CollisionParams, config["collision"])


def planner_params(config, push=False):
    """A push grasp cannot brake the door, which runs away from a hand pushing it fast, so pushing has its own rate."""
    planner = config["planner"]
    rate = planner["push_door_rate_deg"] if push else planner["door_rate_deg"]
    return section_params(
        PlannerParams, planner, goal_angle=0.0, door_rate=np.radians(rate)
    )


def goal_params(config):
    return section_params(
        GoalParams,
        config["goal"],
        workers=config["planner"]["workers"],
        max_sweep=np.radians(config["grasp"]["max_sweep_deg"]),
    )


def scan_params(config):
    return section_params(ScanParams, config["scan"])


def pipeline_rate(today, fast, speed):
    """Half of today at speed 0, today at 0.5, fast at 1."""
    if speed <= 0.5:
        return today * (0.5 + speed)
    return today + (2.0 * speed - 1.0) * (fast - today)


def apply_pipeline_speed(config):
    """The config with every key listed under speed/fast (section: name: fast value) set for speed/pipeline_speed.
    Keys under speed/not_below_today stay at today's value below 0.5."""
    if "speed" not in config:
        return config
    speed = config["speed"]
    level = float(np.clip(speed["pipeline_speed"], 0.0, 1.0))
    scaled = copy.deepcopy(config)
    for section, fast_values in speed["fast"].items():
        for name, fast in fast_values.items():
            key_level = (
                max(level, 0.5) if name in speed.get("not_below_today", []) else level
            )
            scaled[section][name] = scaled_value(
                config[section][name], fast, key_level, name in speed["durations"]
            )
    return scaled


def scaled_value(today, fast, level, is_duration):
    if is_duration:
        return 1.0 / pipeline_rate(1.0 / today, 1.0 / fast, level)
    return pipeline_rate(today, fast, level)
