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
│           ├── tiago_door_localization/  -> AprilTag perception, hinge/handle/doorway poses
│           ├── tiago_move_base_control/  -> move_base wrapper + door approach goal generator
│           ├── tiago_door_bt/            -> py_trees behavior tree executive
│           ├── tiago_door_planning/      -> action server for base + arm trajectory planning
│           ├── tiago_arm_manipulation/   -> MoveIt arm motion planning and gripper control
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
| [tiago_door_localization](ros-ws/src/tiago/tiago_door_localization/README.md) | AprilTag nodes, topics, doorway pose services |
| [tiago_move_base_control](ros-ws/src/tiago/tiago_move_base_control/README.md) | Navigation nodes, goal sender, door approach generator |
| [tiago_door_bt](ros-ws/src/tiago/tiago_door_bt/README.md) | Behavior tree phases and custom BT nodes |
| [tiago_door_planning](ros-ws/src/tiago/tiago_door_planning/README.md) | Planning action server, action definition, parameters |
| [tiago_arm_manipulation](ros-ws/src/tiago/tiago_arm_manipulation/README.md) | Arm services, gripper control, MoveIt parameters |
| [tiago_slam](ros-ws/src/tiago/tiago_slam/README.md) | All SLAM modes, real robot usage, map saving |
| [tiago_vicon](ros-ws/src/tiago/tiago_vicon/README.md) | VICON bridge, position logging, path replay |
