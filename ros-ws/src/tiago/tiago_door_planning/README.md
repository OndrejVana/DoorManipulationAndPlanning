# tiago_door_planning

Opens a door with TIAGo++ (ROS 1 Melodic, Python 3.6): plans a whole-body opening, grasps the handle and tracks
the plan closed-loop with base and arm together.

- Requires: wrist force/torque sensors, latch disabled, interior door 0.8–1.0 m, contact only at the handle.
- No PAL Whole Body Control.

## Quick start

```bash
roslaunch tiago_door_planning door_opening.launch
rviz -d $(rospack find tiago_door_planning)/rviz/door_opening.rviz   # separate terminal
rosrun tiago_door_planning run_open_door.py           # _plan_only:=true, _release_at_end:=true
```

- `run_open_door.py` reads `/door/hinge_pose_map`, `/door/handle_pose_map`, `/door/hinge_side`, `/door/interaction`
  (`tiago_door_localization`): the handle must be in reach and the tag in view.
- The normal way in is the door mission (`tiago_door_mission`), which approaches the door first and passes it after.
- `main.launch` starts the same servers (`use_door_planning`, default true).
- RViz: green path = holding the handle, blue = driving released, orange leaf = believed door angle.

## How an opening runs

1. **Plan:** from the current base pose. When no arm reaches the handle from there, the plan starts released and
   drives onto a grasp spot first. A* per arm over base pose, door angle, grasp state and grip (arm + front/back handle). Checks IK, joint
   limits and collisions on every edge; plans regrasps (with either arm) when one grasp can't open the door far enough.
2. **Choose:** the plan with the fewest grasps, then the shortest.
3. **Grasp:** zero the wrist wrench, approach the handle and stop on contact, close the hand. The laser confirms the
   hand is on the handle.
4. **Track:** 50 Hz base (Kanayama) + arm (damped least squares) tracking. The gripper complies with force along the
   leaf and vertically. Aborts on stall, large gripper error or lost handle (laser).
5. **Regrasp:** release, fold the arm, drive, grasp again; replan when the base or door ends off plan.
6. **Recover:** on a failed segment, let go and replan from the measured state (`goal/recoveries`).

- Goal angle: the passable angle (from `goal/pass_width`) + `planner/passable_margin_deg`. If unreachable, the
  opening settles at the grid angle below, never under the passable angle.
- Ends holding the handle at the goal angle, or with `release_at_end` released and the arm parked (also tried after a
  failure). `pass_door` then drives through.
- Every plan reads the costmap afresh (`costmap/topic`, must be in the door poses' frame). Only lethal cells count,
  checked against the base; the door leaf is modelled, so its swing sector should be masked out of the costmap
  (`DoorMaskLayer`).

## Action servers

All in one node (`door_opening_server.py`), one request at a time (a second one is refused as busy):

- `open_door` (`OpenDoor`): hinge pose, handle pose, hinge side, push, `plan_only`, `release_at_end` -> success, final
  door angle, message. A preempt cancels the running segment.
- `pass_door` (`PassDoor`): door + the angle the opening left it at -> plans a released base path through the gap to
  the first reachable of `pass/distances` past the doorway and tracks it.
- `plan_approach` (`PlanApproach.srv`): door -> base poses to start the opening from: grasp spots for the closed door
  (nearest the middle of the reach ring first), each kept only when a whole opening plans from it, up to
  `goal/approach_plans`, fewest grasps then shortest first. Costs one opening plan per pose.
- `measure_door` (`MeasureDoor.srv`): door -> laser fit of the leaf angle (`seen`, `angle`) and free-space bound.

`execute_door_segment` (`ExecuteDoorSegment`, `door_tracking_server.py`): one grasped or base-only segment, tracked
closed-loop.

## Parameters (`config/door_motion.yaml`)

- `speed`: `pipeline_speed` in [0, 1] (0 = half speed, 0.5 = section values, 1 = fast set).
- `door`, `arm`, `collision`: scene defaults, collision spheres, margins.
- `tracking`: gains, limits, slowdown/stop/abort thresholds.
- `grasp`: arms, grasp offsets per grip, approach and hand settings, sweep per grasp, pull limit.
- `planner`: lattice, door angle step, speeds, joint margins, passable margin.
- `goal`: passable width, reach band, regrasp searches, recoveries.
- `scan`: laser door-angle fit.
- `force`: wrench topic, zeroing, filter, contact stop, tracking compliance, force lost-grasp test (off by default).
- `pass`: distances past the doorway `pass_door` drives to.
- `costmap`: topic (read per plan), wait timeout, occupied threshold.

## Package structure

```text
tiago_door_planning/
├── action/
│   ├── OpenDoor.action            # Full opening: door poses in, success + final door angle out
│   ├── PassDoor.action            # Drive through the opened doorway
│   └── ExecuteDoorSegment.action  # One segment: trajectory in, tracked closed-loop (grasped or base only)
├── srv/
│   ├── PlanApproach.srv           # Base poses to grasp the closed door from
│   └── MeasureDoor.srv            # Laser door angle
├── config/
│   └── door_motion.yaml           # All parameters
├── launch/
│   ├── door_opening.launch        # Tracking server + opening server + plan markers
│   └── door_tracking.launch       # Tracking server only
├── rviz/
│   └── door_opening.rviz          # Door plan markers, laser, door poses
├── scripts/
│   ├── door_opening_server.py     # open_door, pass_door, plan_approach, measure_door
│   ├── door_tracking_server.py    # execute_door_segment action server (50 Hz tracking)
│   ├── run_open_door.py           # Sends the perceived door to open_door and logs the result
│   └── door_plan_viz.py           # RViz markers: walls, leaf, planned paths, stage
├── src/tiago_door_planning/
│   ├── model/
│   │   ├── geometry.py            # SE(2)/SE(3) helpers, box distances
│   │   ├── door_scene.py          # Door model: hinge, leaf, walls, handle pose per angle, swing polygon
│   │   ├── kinematics.py          # Pinocchio arm model: FK, Jacobian, DLS IK, link spheres, joint limits
│   │   └── collision.py           # Base + arm spheres vs door scene and costmap, arm vs arm
│   ├── planning/
│   │   ├── primitives.py          # Lattice base motion primitives
│   │   ├── astar.py               # Plain A*
│   │   ├── door_planner.py        # A* over (x, y, heading, door angle, grasped, grip)
│   │   ├── grips.py               # Grips, grasp offsets, IK checks, plan cost
│   │   ├── door_goal.py           # Passable door angle and goal range
│   │   ├── grasp_spots.py         # Reachable grasp poses and spots after a release
│   │   ├── opening_plan.py        # Plan to the goal angle, regrasps, extension
│   │   ├── pass_plan.py           # Released drive through the opened doorway
│   │   ├── approach.py            # Where to stand to start opening the closed door
│   │   └── plan_output.py         # Planner steps -> resampled segments
│   ├── motion/
│   │   ├── trajectory.py          # DoorTrajectory, reference sampling, split by sweep
│   │   ├── tracking.py            # Kanayama base tracking + DLS arm with null-space terms
│   │   ├── compliance.py          # Admittance offset of the gripper target
│   │   ├── grasp_motion.py        # Grasp offset, approach targets, IK along lines, joint paths
│   │   └── arm_moves.py           # Collision-checked moves to and off the handle
│   ├── sensing/
│   │   ├── door_sensing.py        # Laser door angle, free-space bound, lost-grasp test
│   │   └── wrench.py              # Wrist wrench: zeroing, low-pass, frames, contact and lost-grasp tests
│   └── ros/
│       ├── robot.py               # Joint states, wrench, TF, arm/hand moves, laser
│       ├── config.py              # door_motion.yaml -> parameter tuples
│       └── ros_conversions.py     # ROS message <-> numpy
└── tests/                         # pytest, same folders as src; run with `make test`
```
