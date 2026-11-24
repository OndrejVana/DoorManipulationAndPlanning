# TIAGo++ Door Bringup Package

Main entry point for the TIAGo++ door-opening system: one `main.launch` for sim and robot, plus bag replay.

## Normal run

```bash
roslaunch tiago_door_bringup main.launch                     # real robot (simulation:=false is the default)
roslaunch tiago_door_bringup main.launch simulation:=true \
  gzpose:="-x 1.05 -y -1.5 -z 0 -R 0 -P 0 -Y -1.5708"        # sim, push door in view
# drive around until the map covers the door's surroundings, put the door's tag in the camera's view:
rosservice call /door_mission/start
```

The mission's states and preconditions: [tiago_door_mission](../tiago_door_mission/README.md).

## Launch arguments (`main.launch`)

| Argument | Default | Starts |
| --- | --- | --- |
| `simulation` | `false` | Gazebo world with the doors (`tiago_door_sim`); on the robot it kills PAL's nav nodes when SLAM runs |
| `use_slam` | `true` | `tiago_slam` (RTAB-Map, icp odometry) |
| `use_door_perception` | `true` | AprilTag door perception |
| `use_navigation` | `true` | move_base with the door-masked global costmap the planner reads |
| `use_door_planning` | `true` | `open_door`, `pass_door`, `execute_door_segment`, `plan_approach`, `measure_door` |
| `use_mission` | `true` | door mission node, idle until `/door_mission/start` |
| `mission_dry_run` | `false` | mission plans only (nothing moves) |
| `use_vicon` | `false` | VICON bridge |
| `rviz` | `false` | RViz with `rviz/tiago_door_main.rviz` |
| `world_file`, `robot`, `gui`, `gzpose`, ... | | sim only (`gzpose`: robot spawn pose) |
| `door_config` | two-sided door config | AprilTag configuration YAML |

## Bag replay

`scripts/replay.sh <recording dir> [roslaunch args]` replays a `record_bag.sh` recording through SLAM, perception,
move_base and the door servers. It then starts the mission as a dry run after `mission_delay` s (default 30). That
checks the door lock, the mask, the approach poses and the opening plan on real data.

- The bag's `/tf` is played on `/tf_bag`. The recorded `map → odom` is filtered out, since the replayed SLAM publishes
  its own.
- The URDF saved next to the bags becomes `robot_description`.

```bash
rosrun tiago_door_bringup replay.sh ~/bags/door_planning_test_2026-11-19_15-20-11 rviz:=false
```

## Other

```bash
rosrun key_teleop key_teleop.py /key_vel:=/mobile_base_controller/cmd_vel   # drive manually
rm ~/.ros/rtabmap.db                                                         # drop an old map
```
