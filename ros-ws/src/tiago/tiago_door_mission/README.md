# tiago_door_mission

SMACH state machine that takes the robot from "door in view" to "through the door". It locks the door model from the
AprilTag, checks with the laser that the door is closed, drives to a pose the planner can grasp from, opens the door
with `tiago_door_planning` and drives through.

Only the `smach` core is used: `smach_ros` does not import on Python 3 in this image.

## Running

```bash
roslaunch tiago_door_bringup main.launch            # SLAM, perception, move_base, door servers, mission (idle)
# map the area, put the door (its tag) in the camera's view, then:
rosservice call /door_mission/start
rostopic echo /door_mission/state                   # current state, with the reason when one fails
rostopic echo /door_mission/result                  # succeeded | failed: <reason> | preempted
rosservice call /door_mission/stop                  # preempt: cancels every goal, mask off
```

The mission can be started again after it ends, without relaunching.

## States

```text
CHECK_SYSTEM -> TUCK_ARMS -> LOCK_DOOR -> GO_TO_VIEW -> LOCK_VIEW -> CHECK_CLOSED -> MASK_ON -> PLAN_APPROACH -> NAVIGATE
  -> REFINE_DOOR -> CHECK_CLOSED_NEAR -> OPEN_DOOR -> PASS_DOOR -> MASK_OFF -> succeeded
OPEN_DOOR failed, attempts left -> REOBSERVE -> RECHECK_CLOSED -> MASK_ON (approach again)
any failure -> SAFE_STOP (cancel goals, mask off) -> failed;   ~stop -> ABORT -> preempted
```

| State | What it does | Fails when |
| --- | --- | --- |
| CHECK_SYSTEM | open_door, pass_door, plan_approach, measure_door, move_base, play_motion, head action, TF map→base_footprint, global costmap | anything is missing |
| TUCK_ARMS | play_motion `home` (open_door takes the arms at its start as "home") | the motion fails |
| LOCK_DOOR | points the head at the last seen tag (aimed at the handle, a tag close by leaves the image), skips the samples `door_tag_pose_node` averaged before the head settled, drops outliers, averages the rest. Only places the viewing pose: from afar the tag's orientation can be off by tens of degrees | too few agreeing samples, or spread over `lock/far` (tried `lock/attempts` times) |
| GO_TO_VIEW | move_base to the first reachable of `view/distances` in front of the handle, facing it; skipped when the robot already stands within `view/max_distance` and `view/max_angle_deg` of the tag's normal | no viewing pose reached |
| LOCK_VIEW | locks again to the stricter `lock/near` spreads; the approach and the opening plan on this lock | samples keep disagreeing, or the tag is out of view |
| CHECK_CLOSED | `measure_door` (laser fit of the leaf) | leaf seen open over `closed_tolerance_deg/far` (an unseen leaf passes here) |
| MASK_ON | publishes the swing sector polygon, enables `DoorMaskLayer` on the global costmap; with `mask/slam_leaf` also asks `track_door_leaf` to hide the leaf from the SLAM's scan | - |
| PLAN_APPROACH | `plan_approach`: grasp spots for the closed door from which the whole opening plans, fewest grasps then shortest first | none |
| NAVIGATE | move_base to the first `approach_attempts` poses in turn | all fail |
| REFINE_DOOR | locks the door again from close by to the stricter `lock/near` spreads (the map drifts while driving), up to `lock/attempts` times; keeps the earlier lock only when the tag is out of view; the masks follow a new lock | samples keep disagreeing |
| CHECK_CLOSED_NEAR | as CHECK_CLOSED, to `closed_tolerance_deg/near`, and the leaf must be seen | leaf open or unseen |
| OPEN_DOOR | `open_door` with `release_at_end` (ends released, arm parked) | after `open_attempts` attempts |
| PASS_DOOR | `pass_door`: planned base path through the gap, executed by the tracker | no path, or tracking fails |

`dry_run` (bag replay): no tuck, no head, no move_base (LOCK_VIEW locks from wherever the robot is); OPEN_DOOR only plans (`plan_only`), PASS_DOOR is skipped.

## What the robot and door need before `start`

- SLAM running with the area mapped (`map` frame, `/map` grid in the global costmap).
- The door's tag in the camera's view; the head is not searched.
- The door closed (latch disabled), the robot on the tag's side.
- Wrist F/T sensors publishing (`open_door` refuses without them).

## Files

- `scripts/door_mission_node.py`: node; `~start` / `~stop` (Trigger), `~state` / `~result` (String, latched).
- `src/tiago_door_mission/mission.py`: the state machine over an `io` object (tested with fakes).
- `src/tiago_door_mission/ros_io.py`: ROS side of `io`.
- `src/tiago_door_mission/door_lock.py`: averaging and agreement checks of door samples.
- `config/mission.yaml`: parameters. The node also loads `door` from `tiago_door_planning/config/door_motion.yaml`
  for the swing sector.
- Tests: `tests/` (run by `make test`).
