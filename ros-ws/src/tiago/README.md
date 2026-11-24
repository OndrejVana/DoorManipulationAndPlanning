# TIAGo Development Workspace

ROS 1 (Melodic, Python 3.6) workspace containing custom packages for autonomous door-opening with the TIAGo robot. The system covers perception, SLAM, navigation, door-opening planning and execution, and a SMACH mission on top.

## Package overview

```text
ros-ws/src/tiago/
├── tiago_door_bringup/        # Unified launch entry point for the full system
├── tiago_door_sim/            # Gazebo world with hinged door + AprilTag models
├── tiago_door_localization/   # AprilTag-based door perception (hinge/handle poses)
├── tiago_move_base_control/   # move_base config + DoorMaskLayer costmap plugin
├── tiago_door_mission/        # SMACH mission: lock door, approach, open, drive through
├── tiago_door_planning/       # Door opening: grasp, whole-body A* plan, closed-loop tracking, regrasp
├── tiago_slam/                # SLAM implementations (slam_toolbox, RTAB-Map, ORB-SLAM2/3)
└── tiago_vicon/               # VICON bridge, position logger, path record/replay
```

## Quick start

Inside the dev container (`cd ros-ws/docker && make up && make build && make shell`), sourcing is automatic:

```bash
# Full simulation run (door in view at start), then start the mission
roslaunch tiago_door_bringup main.launch simulation:=true \
  gzpose:="-x 1.05 -y -1.5 -z 0 -R 0 -P 0 -Y -1.5708"
rosservice call /door_mission/start

# Headless (no Gazebo GUI)
roslaunch tiago_door_bringup main.launch simulation:=true gui:=false
```

## Building

`make build` from `ros-ws/docker` (catkin build with `-DPYTHON_EXECUTABLE=/usr/bin/python3`, extending `/py3_ws`).

## System architecture

```text
tiago_door_sim          -> Gazebo simulation (world + robot)
tiago_slam              -> Map + localization (laser/visual SLAM)
tiago_door_localization -> AprilTag → hinge/handle poses
tiago_move_base_control -> move_base + global costmap with the door swing sector masked
tiago_door_planning     -> open_door / pass_door / plan_approach / measure_door: grasp, plan, track, regrasp
tiago_door_mission      -> SMACH: lock door → approach → open → drive through
tiago_vicon             -> Ground truth (VICON) + position logging + path replay
tiago_door_bringup      -> Orchestrates all of the above via main.launch
```

## Dependencies

| Category | Packages |
| --- | --- |
| Navigation | `move_base`, `amcl`, `map_server`, `navfn`, `base_local_planner` |
| SLAM | `slam_toolbox`, `rtabmap_ros`, `orb_slam2_ros`, `orb_slam3_ros` |
| Perception | `apriltag_ros`, `tf2_ros`, `tf2_geometry_msgs` |
| Manipulation | `pinocchio`, `control_msgs`, `play_motion_msgs`, `actionlib` |
| Mission | `smach` (core only) |
| Simulation | `gazebo_ros`, PAL TIAGo public stack |

## Selected package READMEs

- [tiago_door_bringup](tiago_door_bringup/README.md) — full system launch guide
- [tiago_slam](tiago_slam/README.md) — all SLAM modes and real robot setup
- [tiago_door_localization](tiago_door_localization/README.md) — AprilTag perception pipeline
- [tiago_move_base_control](tiago_move_base_control/README.md) — move_base + door mask layer
- [tiago_door_mission](tiago_door_mission/README.md) — mission states and preconditions
- [tiago_door_planning](tiago_door_planning/README.md) — door opening action servers
- [tiago_door_sim](tiago_door_sim/README.md) — Gazebo simulation setup
- [tiago_vicon](tiago_vicon/README.md) — VICON bridge and position logging
