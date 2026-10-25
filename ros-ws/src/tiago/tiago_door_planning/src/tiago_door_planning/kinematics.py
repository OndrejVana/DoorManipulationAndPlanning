import collections
import numpy as np
import pinocchio as pin
from .geometry import Sphere

ArmModel = collections.namedtuple(
    "ArmModel",
    [
        "model",
        "data",
        "side",
        "gripper_frame_id",
        "sphere_frame_ids",
        "sphere_radius",
        "sphere_spacing",
    ],
)


def arm_joint_names(side):
    return ["arm_%s_%d_joint" % (side, i) for i in range(1, 8)]


def _reference_configuration(model, fixed_positions):
    q = pin.neutral(model)
    for name, value in fixed_positions.items():
        if model.existJointName(name):
            joint = model.joints[model.getJointId(name)]
            q[joint.idx_q] = value
    return q


def _joints_to_lock(model, free_joint_names):
    free_ids = {model.getJointId(name) for name in free_joint_names}
    return [
        joint_id for joint_id in range(1, model.njoints) if joint_id not in free_ids
    ]


def _frame_id(model, name):
    if not model.existFrame(name):
        raise ValueError("Frame %r not found in the robot model" % name)
    return model.getFrameId(name)


def load_arm_model(
    urdf_xml,
    side,
    gripper_frame,
    sphere_links,
    sphere_radius,
    sphere_spacing,
    fixed_positions,
):
    full = pin.buildModelFromXML(urdf_xml)
    q_reference = _reference_configuration(full, fixed_positions)
    model = pin.buildReducedModel(
        full, _joints_to_lock(full, arm_joint_names(side)), q_reference
    )
    sphere_frame_ids = [
        _frame_id(model, name.format(side=side)) for name in sphere_links
    ]
    sphere_frame_ids.append(_frame_id(model, gripper_frame.format(side=side)))
    return ArmModel(
        model,
        model.createData(),
        side,
        sphere_frame_ids[-1],
        sphere_frame_ids,
        sphere_radius,
        sphere_spacing,
    )


def _update_frames(arm, q):
    pin.forwardKinematics(arm.model, arm.data, q)
    pin.updateFramePlacements(arm.model, arm.data)


def gripper_pose(arm, q):
    _update_frames(arm, q)
    return arm.data.oMf[arm.gripper_frame_id].homogeneous.copy()


def gripper_jacobian(arm, q):
    return pin.computeFrameJacobian(
        arm.model, arm.data, q, arm.gripper_frame_id, pin.LOCAL
    )


def joint_limits(arm):
    return arm.model.lowerPositionLimit.copy(), arm.model.upperPositionLimit.copy()


def velocity_limits(arm):
    return arm.model.velocityLimit.copy()


def clamp_to_limits(arm, q):
    lower, upper = joint_limits(arm)
    return np.clip(q, lower, upper)


def _gripper_error(arm, q, target_se3):
    _update_frames(arm, q)
    return pin.log6(arm.data.oMf[arm.gripper_frame_id].actInv(target_se3)).vector


def _ik_step(arm, q, error, damping):
    jacobian = gripper_jacobian(arm, q)
    regularized = jacobian @ jacobian.T + damping**2 * np.eye(6)
    return jacobian.T @ np.linalg.solve(regularized, error)


def solve_ik(arm, target, seed, iterations=200, tolerance=1e-4, damping=1e-2):
    """Return (q, converged) for a gripper target given in the base frame."""
    target_se3 = pin.SE3(target)
    q = clamp_to_limits(arm, np.asarray(seed, dtype=float))
    for _ in range(iterations):
        error = _gripper_error(arm, q, target_se3)
        if np.linalg.norm(error) < tolerance:
            return q, True
        q = clamp_to_limits(arm, q + _ik_step(arm, q, error, damping))
    return q, False


def _points_along_segment(start, end, spacing):
    count = max(1, int(np.ceil(np.linalg.norm(end - start) / spacing)))
    return [start + (end - start) * i / count for i in range(count)]


def link_spheres(arm, q):
    """Collision spheres in the base frame, spaced along the chain of sphere frames."""
    _update_frames(arm, q)
    origins = [
        arm.data.oMf[frame_id].translation.copy() for frame_id in arm.sphere_frame_ids
    ]
    spheres = []
    last_segment = len(origins) - 2
    for index, (start, end) in enumerate(zip(origins[:-1], origins[1:])):
        for center in _points_along_segment(start, end, arm.sphere_spacing):
            spheres.append(Sphere(center, arm.sphere_radius, index == last_segment))
    spheres.append(Sphere(origins[-1], arm.sphere_radius, True))
    return spheres
