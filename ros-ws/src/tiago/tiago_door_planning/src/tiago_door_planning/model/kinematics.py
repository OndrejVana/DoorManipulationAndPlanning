import collections
import numpy as np
import pinocchio as pin
from scipy.linalg.lapack import dgesv
from .geometry import Sphere

ArmModel = collections.namedtuple(
    "ArmModel",
    "model data side gripper_frame_id sphere_frame_ids sphere_radius sphere_spacing",
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


def has_joint_margin(arm, q, margin):
    lower, upper = joint_limits(arm)
    return bool(np.all(q - lower >= margin) and np.all(upper - q >= margin))


def middle_configuration(arm):
    lower, upper = joint_limits(arm)
    return (lower + upper) / 2.0


def clamp_to_limits(arm, q):
    lower, upper = joint_limits(arm)
    return np.clip(q, lower, upper)


def _gripper_error(arm, q, target_se3):
    _update_frames(arm, q)
    return pin.log6(arm.data.oMf[arm.gripper_frame_id].actInv(target_se3)).vector


EYE6 = np.eye(6)
STALL_ITERATIONS = 15
STALL_IMPROVEMENT = 0.99


def _ik_step(arm, q, error, damping):
    jacobian = gripper_jacobian(arm, q)
    regularized = jacobian @ jacobian.T + damping**2 * EYE6
    return jacobian.T @ dgesv(regularized, error)[2]


def solve_ik(arm, target, seed, iterations=200, tolerance=1e-4, damping=1e-2):
    """Return (q, converged) for a gripper target given in the base frame. Gives up early once the error has not
    shrunk by 1 % for STALL_ITERATIONS steps, which is where unreachable targets end up.
    """
    target_se3 = pin.SE3(target)
    lower, upper = arm.model.lowerPositionLimit, arm.model.upperPositionLimit
    q = np.clip(np.asarray(seed, dtype=float), lower, upper)
    best, best_iteration = np.inf, 0
    for iteration in range(iterations):
        error = _gripper_error(arm, q, target_se3)
        size = np.sqrt(error @ error)
        if size < tolerance:
            return q, True
        if size < best * STALL_IMPROVEMENT:
            best, best_iteration = size, iteration
        elif iteration - best_iteration >= STALL_ITERATIONS:
            break
        q = np.clip(q + _ik_step(arm, q, error, damping), lower, upper)
    return q, False


def link_spheres(arm, q):
    """Collision spheres in the base frame, spaced along the chain of sphere frames; the last segment and the last
    frame belong to the gripper."""
    _update_frames(arm, q)
    origins = np.array(
        [arm.data.oMf[frame_id].translation for frame_id in arm.sphere_frame_ids]
    )
    spheres = []
    for index, (start, delta) in enumerate(
        zip(origins[:-1], origins[1:] - origins[:-1])
    ):
        count = max(1, int(np.ceil(np.linalg.norm(delta) / arm.sphere_spacing)))
        centers = start + delta * np.arange(count, dtype=float)[:, None] / count
        spheres += [
            Sphere(center, arm.sphere_radius, index == len(origins) - 2)
            for center in centers
        ]
    spheres.append(Sphere(origins[-1], arm.sphere_radius, True))
    return spheres
