import collections
import numpy as np

from ..model.geometry import wrap_angle
from ..motion.trajectory import SAMPLE_PERIOD, DoorTrajectory

Segment = collections.namedtuple("Segment", "grasped trajectory face")


def split_by_grasp(steps):
    runs = []
    for step in steps[1:]:
        if runs and runs[-1][0] == step.grasped:
            runs[-1][1].append(step)
        else:
            runs.append((step.grasped, [step]))
    return runs


def timed_samples(steps):
    times, poses, angles = [], [], []
    elapsed = 0.0
    for step in steps:
        fractions = np.linspace(0.0, 1.0, len(step.poses) + 1)[1:]
        times += list(elapsed + fractions * step.duration)
        poses += list(step.poses)
        angles += list(step.angles)
        elapsed += step.duration
    return np.array(times), np.array(poses), np.array(angles)


def resample(times, values, sample_times):
    return np.array([np.interp(sample_times, times, values[:, i]) for i in range(values.shape[1])]).T


def uniform_times(duration):
    times = np.arange(0.0, duration + 1e-9, SAMPLE_PERIOD)
    return times if times[-1] >= duration - 1e-9 else np.append(times, duration)


def ramped_times(duration, ramp):
    """Plan times at evenly spaced samples of a motion slowed linearly to a stop over its last ramp seconds, which
    then take twice as long."""
    ramp = min(ramp, duration)
    if ramp <= 0.0:
        return uniform_times(duration)
    start = duration - ramp
    times = uniform_times(duration + ramp)
    into_ramp = np.maximum(times - start, 0.0)
    return np.minimum(times, start) + into_ramp - into_ramp**2 / (4.0 * ramp)


def resample_segment(start_pose, start_angle, steps, ramp=0.0):
    times, poses, angles = timed_samples(steps)
    times = np.concatenate([[0.0], times])
    poses = np.vstack([start_pose, poses])
    poses[:, 2] = np.unwrap(poses[:, 2])
    angles = np.concatenate([[start_angle], angles])
    sample_times = ramped_times(times[-1], ramp)
    sampled_poses = resample(times, poses, sample_times)
    sampled_poses[:, 2] = [wrap_angle(yaw) for yaw in sampled_poses[:, 2]]
    return sampled_poses, np.interp(sample_times, times, angles), sample_times


def step_end_times(steps):
    return np.concatenate([[0.0], np.cumsum([step.duration for step in steps])])


def resample_arm(start_q, steps, sample_times):
    """The planned arm configuration at each sample; the planner fixes it at the end of every step."""
    qs = np.vstack([start_q] + [step.q for step in steps])
    return resample(step_end_times(steps), qs, sample_times)


def is_regrasp(step):
    return len(step.poses) == 1


def plan_to_segments(steps, drive_ramp=0.0, held_ramp=0.0):
    """One segment per run of motion steps; release and grasp happen between segments.

    A released segment's grip is the one that let go; its drive slows to a stop over its last drive_ramp seconds, so
    the base ends where it was planned to rather than lagging behind a reference that stopped at full speed. A held
    segment slows down over its last held_ramp seconds, so a door the hand cannot brake does not swing on far.
    """
    segments = []
    previous = steps[0]
    for grasped, run in split_by_grasp(steps):
        regrasps = [step for step in run if is_regrasp(step)]
        motion = [step for step in run if not is_regrasp(step)]
        if regrasps and regrasps[0] is run[0]:
            previous = regrasps[0]
        if motion:
            poses, angles, sample_times = resample_segment(
                previous.poses[-1],
                previous.angles[-1],
                motion,
                held_ramp if grasped else drive_ramp,
            )
            arm_qs = resample_arm(np.array(previous.q), motion, sample_times)
            grip = run[-1].grip
            trajectory = DoorTrajectory(poses, angles, SAMPLE_PERIOD, grip.arm, arm_qs)
            segments.append(Segment(grasped, trajectory, grip.face))
        previous = run[-1]
    return segments
