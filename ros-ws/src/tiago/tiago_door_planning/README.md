# tiago_door_planning

ROS 1 Melodic (Python 3.6) package that opens a door with TIAGo++. Given the perceived hinge and handle poses, it
plans a whole-body opening with A* for each arm, grasps the handle with the arm whose plan needs the fewest grasps,
and tracks the plan closed-loop (base and arm together). It regrasps the handle when one grasp cannot sweep the whole
opening, also with the other arm or after driving the base with the hand off the door. A pull door is pulled to
`grasp/pull_limit_deg` and then pushed on from its far side by the handle on the other face. The laser measures the
door angle on its own, so a hand that slipped off the handle stops the opening instead of reporting success, and a
door it cannot see is bounded by the laser rays that pass where the leaf would be.

Constraints it is designed for: no PAL Whole Body Control, no force/torque sensor, latch disabled, interior doors
0.8–1.0 m wide, robot touches the door only at the handle.

---

## Package structure

```text
tiago_door_planning/
├── action/
│   ├── OpenDoor.action            # Full opening: door poses in, success + final door angle out
│   └── ExecuteDoorSegment.action  # One segment: trajectory in, tracked closed-loop (grasped or base only)
├── config/
│   └── door_motion.yaml           # All parameters (door, arm spheres, collision, tracking, grasp, planner, goal)
├── launch/
│   ├── door_opening.launch        # Tracking server + opening server
│   └── door_tracking.launch       # Tracking server only
├── scripts/
│   ├── door_opening_server.py     # open_door action server
│   ├── door_tracking_server.py    # execute_door_segment action server (50 Hz whole-body or base-only tracking)
│   └── run_open_door.py           # Sends the perceived door to open_door and logs the result
├── src/tiago_door_planning/
│   ├── model/                     # What the robot and the door are
│   │   ├── geometry.py            # SE(2)/SE(3) helpers, box distances
│   │   ├── door_scene.py          # Door model from perception: hinge, leaf, walls, handle pose per angle
│   │   ├── kinematics.py          # Pinocchio arm model: FK, Jacobian, DLS IK, link spheres, joint limits
│   │   └── collision.py           # Base + arm spheres vs door scene and costmap, arm vs arm
│   ├── planning/                  # How to open the door
│   │   ├── primitives.py          # Lattice-exact base motion primitives
│   │   ├── astar.py               # Plain A*
│   │   ├── door_planner.py        # A* over (x, y, heading, door angle, grasped, grip = arm + handle face)
│   │   ├── grips.py               # Grips, their grasp offsets and IK checks for the planner, plan cost
│   │   ├── door_goal.py           # Passable door angle and the cap where grasp poses run out
│   │   ├── grasp_spots.py         # Reachable grasp poses, grasp spots after a release, drive distance map
│   │   ├── opening_plan.py        # Plan to the passable angle (driving to a new grasp after a release), then extend
│   │   └── plan_output.py         # Planner steps -> resampled segments
│   ├── motion/                    # How the robot follows the plan
│   │   ├── trajectory.py          # DoorTrajectory, reference sampling, split by sweep
│   │   ├── tracking.py            # Control law: Kanayama base tracking + DLS arm with null-space terms
│   │   ├── grasp_motion.py        # Grasp offset, approach/lift targets, IK along lines, joint paths
│   │   └── arm_moves.py           # Collision-checked moves to and off the handle
│   ├── sensing/
│   │   └── door_sensing.py        # Door angle fitted to laser points on the leaf, free-space bound, lost-grasp test
│   └── ros/                       # ROS glue
│       ├── robot.py               # Joint states, TF, arm models, arm/hand moves, laser reading
│       ├── config.py              # door_motion.yaml -> parameter tuples
│       └── ros_conversions.py     # ROS message <-> numpy
└── tests/                         # pytest, same folders as src (kinematics tests skip without pinocchio)
```

---

## How an opening runs

1. **Plan, per arm in `grasp/arms`.** Plan an arm motion to the handle (approach from in front; lift from below
   when the robot stands too close), then A* from that grasp over base pose, door angle, grasp state and grip. A
   grip is an arm on one handle: the front handle the robot faces, and for a pull door also the back handle
   (mirrored through the leaf), which pushes the door on from its far side. Front grips of a pull door stop at
   `grasp/pull_limit_deg`. Door angles after the start lie on the `planner/angle_step_deg` grid, so a replan from a
   door pushed a little during the grasp aims at the same angles. Arm feasibility is IK along every edge, with no joint moving more than
   `planner/max_joint_step` between samples so the tracker can follow the arm, and grasped edges last at least the
   joint travel over `planner/arm_speed`; collisions cover the door frame, wall and swinging
   leaf (only the gripper may touch the door, at the handle) and the two arms against each other. The arm not
   holding the door waits at its starting (home) configuration. When one grasp runs out of reach or hits its limit,
   the plan lets go, finds base poses (grasp spots) where some grip grasps and can move the door further, drives
   onto the nearest reachable one and searches on from the new grasp, up to `goal/regrasp_searches` times.
2. **Choose.** Keep the plan with the fewest grasps, then the shortest one.
3. **Grasp.** Execute the grasp motion (checked with the smaller `grasp/move_margin`; arm moves are timed by
   `grasp/move_speed` and `grasp/approach_speed`, the hand opens on the way; an approach the handle stops within
   `grasp/contact_tolerance` of its end is done), close the hand and take the door angle
   from the hand on the handle. The laser reads a turned door a few degrees off, so it only confirms that the hand holds
   the handle: a laser angle more than `grasp/door_agreement_deg` away from the hand's means the grasp missed (a
   released door can drift a few degrees). Where the
   laser does not see the leaf (a pushed door past ~75° in the sim), the rays that pass where the leaf would be give the
   angle the door is open at least; a door seen open further than the hand means the hand closed on air, and the door
   angle becomes that bound.
4. **Track.** Grasped segments are cut into pieces of at most `grasp/max_sweep_deg` of door motion and sent to
   `execute_door_segment`: the base follows its reference with a Kanayama law (plus `tracking/k_heading`, which turns it onto the
   reference heading while progress stands still), the arm keeps the gripper on the
   handle with damped least squares around the planned joint velocity, its null space pulled to the planned arm
   configuration, and rotation about the bar left free because the wrapped hand rolls on it (joint commands are integrated, at most `tracking/max_command_lead` ahead of the
   arm, so a joint sagging under load still moves), and progress slows down or stops on large errors. The door is passive, so
   progress jumps ahead when the measured door angle leads the reference, as far as the base can follow
   (`tracking/base_error_slow`); the gripper target follows the door meanwhile. The laser door angle (`scan`) runs alongside:
   when the hand runs `tracking/lost_grasp_deg` ahead of the door, or the door that far ahead of the hand (a push that
   kicked it away), for `tracking/lost_grasp_s`, the segment aborts.
5. **Regrasp.** Between pieces the hand opens and leaves the handle: it retraces its approach while the arm is still
   within `grasp/retrace_tolerance` of the grasp; otherwise a hook drops `grasp/release_drop` and a push grasp backs
   away from the door by as much; a hand already closer to
   the leaf than the margin may leave as long as it does not close in further. A released segment folds the arm home (stepping the base
   straight away from the leaf by `grasp/standoff` first when the arm cannot fold where it stands) and drives the base
   alone, so the planner's released robot carries a tucked arm; the next grasp unfolds through `grasp/unfold_q`. The
   last step onto a grasp spot turns, drives straight and turns, or glides at most 2 cm sideways, since the base cannot
   step sideways. After the release the laser checks the leaf, which a push grasp cannot brake: when it swung on by more
   than `grasp/door_agreement_deg` the rest is planned again for the door where it stopped; the rest of the
   opening is planned again from the grasp after the drive when the base ends more than `goal/replan_base_error` or
   `goal/replan_heading_error_deg` off the planned grasp pose (otherwise the tracker takes up the difference); that plan
   uses the grasp position measured on the handle with the planned rotation, and a grasp more than
   `grasp/max_grasp_error` off is tried again (`grasp/grasp_attempts`). When that plan finds nothing (the search is
   sensitive to where it starts), the robot follows the plan it already has. A piece that starts after the door has
   moved a little keeps the base motion planned at its first door angle. The grasp IK starts from the arm configuration the plan grasps with, since the opening
   may not continue from another solution. A grasp with the other arm first
   sends the arm that let go home.
6. **Recover.** When a segment stalls or loses the handle, an arm move fails, or no grasp or plan is found after a
   regrasp, the robot lets go and plans the rest of the opening from
   where it and the door really are, up to `goal/recoveries` times. A failure with the door already past the passable
   angle (e.g. the leaf against its stop) ends the opening as a success; a plan that ends short of it counts as a
   failed segment.

No new grasp is made once the door is at the passable angle, since each costs a release, a drive and a regrasp. The
door angle that decides it comes from the hand, which reads the sim door a degree or two low, rather than from the
laser fit, which reads it about 2.5° high. The opening ends holding the handle (or with the hand on air) at the
passable angle or wider; passing through is left to the caller. The scene model puts
the hinge axis on the swing-side face of the leaf and mirrors the back handle through the middle of the leaf, so
`door/thickness` should match the real door.

---

## Quick start

```bash
# from ros-ws/docker
make build
make shell
roslaunch tiago_door_planning door_opening.launch
rosrun tiago_door_planning run_open_door.py _push:=false
```

`run_open_door.py` reads `/door/hinge_pose_map`, `/door/handle_pose_map` and `/door/hinge_side` (published by
`tiago_door_localization`). `main.launch use_door_planning:=true` starts the same servers.

---

## Action servers

### `open_door` — `tiago_door_planning/OpenDoor`

```text
# Goal
geometry_msgs/PoseStamped hinge_pose
geometry_msgs/PoseStamped handle_pose   # z axis = door normal
string hinge_side                       # "left" | "right"
bool push
---
# Result
bool success
float64 final_door_angle
string message
---
# Feedback
string stage                            # planning with an arm, grasping, opening to X deg, releasing, driving
```

### `execute_door_segment` — `tiago_door_planning/ExecuteDoorSegment`

Goal: door poses, arm, grasped flag, base poses and door angles sampled every `sample_period` (0.1 s), the planned arm configuration at every sample (`arm_qs`, row-major). Result:
success, final door angle, message. Feedback: progress, gripper error, base error, measured door angle. A grasped
segment aborts when the gripper error exceeds `tracking/gripper_error_abort`; any segment aborts when progress stalls
for `tracking/stall_timeout_s`, or when the laser shows the door lagging behind the hand or running ahead of it. A
segment ends early when the laser shows the door open at least to its last angle. The final door angle is the hand's
(the planned one for a released segment), unless the laser shows the hand ahead of the leaf (then the laser fit) or
the door more than `tracking/lost_grasp_deg` further open (then that bound, which reads high like the fit while the
leaf is in view). The leaf fit and the bound only consider angles from the segment's first door angle
less `tracking/lost_grasp_deg` up, since a door without a spring does not close on its own and the frame and wall
line up with a closed leaf. A released segment moves the base only and leaves the arm where it is.

---

## Parameters (`config/door_motion.yaml`)

| Section | Contents |
|---|---|
| `door` | Default leaf width/thickness/height and wall size for the scene model |
| `arm` | Gripper frame and the links approximated by collision spheres |
| `collision` | Base radius and clearance margin |
| `tracking` | Rate, base gains and limits, arm gains, joint-limit margin, slowdown/stop/abort thresholds, lost-grasp test |
| `grasp` | Arms to plan with; for the pull grasp (hook under the bar) and the push grasp (palm on the bar, fingers over it) the orientation and the gripper offset from the handle per arm in door terms (toward hinge, down, out of the handle's face); hand open/closed positions, sweep per grasp, pull limit before switching to the back handle (90 disables it), release drop, retrace tolerance and how far short of the handle an approach stopped by contact may end, clearance margin and joint speeds for the moves to and from the handle, IK settings |
| `planner` | Lattice resolution, primitives, door angle step, base, door and arm speeds used for edge durations, expansion and stall budgets, largest joint step between samples, joint-limit margin (larger for push grips, whose wrist runs toward its limits), released arm configuration |
| `goal` | Preferred and passable opening, grasp-pose reach band, searches after a release, recoveries after a failed segment, base error that triggers a replan after a regrasp |
| `scan` | Leaf fit and free-space bound (`max_rays_through` rays may cross a leaf) for the independent door angle |
| `costmap` | Optional occupancy grid used for base collisions |

---

## Tests

```bash
# from ros-ws/docker
make test
```
