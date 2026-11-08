import numpy as np

from ..model.collision import arms_collide
from ..model.door_scene import (
    back_scene,
    handle_pose,
    rotation_about_hinge,
    scene_from_door_poses,
)
from ..model.geometry import invert, pose2d_to_matrix, rpy_to_rotation
from ..model.kinematics import (
    has_joint_margin,
    link_spheres,
    middle_configuration,
    solve_ik,
)
from ..motion.arm_moves import Surroundings, grasp_reachable
from ..motion.grasp_motion import grasp_offset, toward_robot
from ..motion.trajectory import split_by_sweep
from ..ros.config import collision_params, door_dimensions
from .door_planner import ArmChecks, Grip


def other_side(side):
    return "left" if side == "right" else "right"


def grasp_action(push, face):
    """A pull door is pushed on by its back handle."""
    return "push" if push or face == "back" else "pull"


def grasp_rotation(config, action):
    """Gripper orientation in the handle frame for a pull or push grasp."""
    return rpy_to_rotation(*config["grasp"][action]["rpy"])


def door_scenes(hinge_pose, handle_pose, hinge_side, push, config):
    """The front scene, plus the back scene when a pull door is pushed on from its far side."""
    scene = scene_from_door_poses(
        hinge_pose, handle_pose, hinge_side, push, door_dimensions(config)
    )
    if push or config["grasp"]["pull_limit_deg"] >= 90.0:
        return {"front": scene}
    return {"front": scene, "back": back_scene(scene)}


def face_normals(scenes, base_pose):
    """Normals of the closed door out of each handle's face; the back face looks away from the robot."""
    front = toward_robot(scenes["front"], 0.0, base_pose[:2])
    return {"front": front, "back": -front}


def all_grips(config, scenes):
    return [Grip(side, face) for side in config["grasp"]["arms"] for face in scenes]


def segment_grip(segment):
    return Grip(segment.trajectory.side, segment.face)


def grasp_offsets(config, push, scenes, normals):
    offsets = {}
    for grip in all_grips(config, scenes):
        action = grasp_action(push, grip.face)
        door_offset = config["grasp"][action]["door_offset"][grip.arm]
        offsets[grip] = grasp_offset(
            scenes[grip.face],
            normals[grip.face],
            grasp_rotation(config, action),
            door_offset,
        )
    return offsets


def grip_checks(config, push, scenes, offsets, normals, arms, home, grip):
    """IK on the handle of the grip's face, with the other arm waiting at home. Pull grips stop at the pull limit when
    the far side of the door is there to push on; pushing runs the wrist toward its limits as the door turns, so a
    push grip keeps more joint margin. A regrasp also needs the way in from the folded arm (reachable).
    """
    arm, scene, normal = arms[grip.arm], scenes[grip.face], normals[grip.face]
    passive_spheres = link_spheres(
        arms[other_side(grip.arm)], home[other_side(grip.arm)]
    )
    planner, collision = config["planner"], collision_params(config)
    pushing = grasp_action(push, grip.face) == "push"
    joint_margin = planner["push_joint_margin"] if pushing else planner["joint_margin"]
    max_angle = (
        np.radians(config["grasp"]["pull_limit_deg"])
        if grip.face == "front" and "back" in scenes
        else np.inf
    )
    unfold_q = np.array(config["grasp"]["unfold_q"])
    default_seeds = (unfold_q, middle_configuration(arm))
    handles = {}
    accepted = {"q": None, "spheres": None}

    def handle_at(door_angle):
        """The handle pose and its face normal at door_angle; the planner asks for the same angles many times."""
        if door_angle not in handles:
            face_normal = rotation_about_hinge(scene, door_angle)[:2, :2] @ normal
            handles[door_angle] = (handle_pose(scene, door_angle), face_normal)
        return handles[door_angle]

    def solve(base_pose, door_angle, seed):
        """Only from in front of the handle's face. Without a seed the IK starts from unfold_q, which the arm unfolds
        through, so the solution lies where the arm can get to, and then from the middle of the joint ranges.
        """
        if door_angle > max_angle + 1e-9:
            return None
        handle, face_normal = handle_at(door_angle)
        if (base_pose[:2] - handle[:2, 3]) @ face_normal <= 0.0:
            return None
        target = invert(pose2d_to_matrix(base_pose)) @ handle @ offsets[grip]
        for start in default_seeds if seed is None else (seed,):
            q, converged = solve_ik(
                arm, target, start, iterations=planner["ik_iterations"]
            )
            if converged and has_joint_margin(arm, q, joint_margin):
                active = link_spheres(arm, q)
                if not arms_collide(active, passive_spheres, collision):
                    accepted["q"], accepted["spheres"] = q, active
                    return q
        return None

    def spheres(q):
        """The planner checks collisions right after solve accepted q, so its spheres are reused."""
        active = accepted["spheres"] if q is accepted["q"] else link_spheres(arm, q)
        return active + passive_spheres

    move_collision = collision._replace(margin=config["grasp"]["move_margin"])

    def reachable(base_pose, door_angle, q):
        around = Surroundings(scene, door_angle, base_pose, None, passive_spheres)
        return grasp_reachable(
            arm,
            around,
            offsets[grip],
            home[grip.arm],
            q,
            unfold_q,
            config["grasp"],
            move_collision,
        )

    return ArmChecks(solve, spheres, max_angle, reachable)


def arm_checks(config, push, scenes, offsets, normals, arms, home):
    return {
        grip: grip_checks(config, push, scenes, offsets, normals, arms, home, grip)
        for grip in all_grips(config, scenes)
    }


def plan_cost(segments, max_sweep):
    """Grasps first, then length."""
    grasps = sum(
        len(split_by_sweep(s.trajectory, max_sweep)) for s in segments if s.grasped
    )
    return grasps, sum(len(s.trajectory.door_angles) for s in segments)


def describe_segment(segment):
    angles = np.degrees(segment.trajectory.door_angles)
    action = (
        "holds the %s handle" % segment.face if segment.grasped else "released, drives"
    )
    return "%s %s %.0f-%.0f deg" % (
        segment.trajectory.side,
        action,
        angles[0],
        angles[-1],
    )
