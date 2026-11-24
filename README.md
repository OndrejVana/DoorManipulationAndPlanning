# TIAGO Thesis

This repository contains software developed for the PAL Robotics TIAGo mobile manipulator, focused on autonomous door opening. It integrates perception, mapping, base and arm motion planning, and control into a modular ROS-based architecture that enables the robot to detect a door, localize its handle and hinge, plan coordinated whole-body motions, and execute the opening maneuver. The system is designed for reliable operation in both simulation and real-world environments, with clear separation between perception, planning, and control components.

## Quick start

```bash
cd ros-ws/docker && make up && make build && make shell
roslaunch tiago_door_bringup main.launch
```

Details in [ros-ws/README.md](ros-ws/README.md). Agent instructions in [CLAUDE.md](CLAUDE.md).

## Hierarchy

```text
TIAGO_thesis/
├── README.md
│
├── CLAUDE.md                             -> agent instructions (.claude/ holds skills)
├── ROBOT_TIME_SYNC.md                    -> chrony time sync with the robot
│
├── ros-ws/                               -> ROS workspace
│   │   README.md                         -> Docker / make usage
│   ├── docker/                           -> Dockerfile, compose, Makefile
│   └── src/tiago/
│           README.md                     -> package overview & system architecture
│           ├── tiago_door_bringup/       -> unified launch entry point for the full system
│           ├── tiago_door_sim/           -> Gazebo world with hinged door + AprilTag models
│           ├── tiago_door_localization/  -> AprilTag perception, hinge/handle poses
│           ├── tiago_move_base_control/  -> move_base config + DoorMaskLayer costmap plugin
│           ├── tiago_door_mission/       -> SMACH mission: lock door, approach, open, drive through
│           ├── tiago_door_planning/      -> door opening: grasp, plan, closed-loop tracking, regrasp
│           ├── tiago_slam/               -> SLAM implementations (slam_toolbox, RTAB-Map, ORB-SLAM)
│           └── tiago_vicon/              -> VICON bridge, position logger, path record/replay
│
└── docs/                                 -> All documents connected to the work
```

| README | Description |
| --- | --- |
| [ros-ws/README.md](ros-ws/README.md) | Docker + make usage, prerequisites |
| [ros-ws/src/tiago/README.md](ros-ws/src/tiago/README.md) | All ROS packages, system architecture, quick start |
| [tiago_door_bringup](ros-ws/src/tiago/tiago_door_bringup/README.md) | Full system launch, component flags, launch arguments |
| [tiago_door_sim](ros-ws/src/tiago/tiago_door_sim/README.md) | Gazebo simulation setup and launch arguments |
| [tiago_door_localization](ros-ws/src/tiago/tiago_door_localization/README.md) | AprilTag nodes and topics |
| [tiago_move_base_control](ros-ws/src/tiago/tiago_move_base_control/README.md) | move_base config, DoorMaskLayer costmap plugin |
| [tiago_door_mission](ros-ws/src/tiago/tiago_door_mission/README.md) | Mission states, preconditions, how to run |
| [tiago_door_planning](ros-ws/src/tiago/tiago_door_planning/README.md) | Door opening action servers, actions, parameters |
| [tiago_slam](ros-ws/src/tiago/tiago_slam/README.md) | All SLAM modes, real robot usage, map saving |
| [tiago_vicon](ros-ws/src/tiago/tiago_vicon/README.md) | VICON bridge, position logging, path replay |
