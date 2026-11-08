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


def planner_params(config):
    return section_params(PlannerParams, config["planner"], goal_angle=0.0)


def goal_params(config):
    return section_params(
        GoalParams, config["goal"], workers=config["planner"]["workers"]
    )


def scan_params(config):
    return section_params(ScanParams, config["scan"])
