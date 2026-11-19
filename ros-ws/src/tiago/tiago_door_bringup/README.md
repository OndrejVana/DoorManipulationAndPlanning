# TIAGo++ Door Bringup Package

Main entry point for launching the complete TIAGo++ dual-arm door navigation system.

## Overview

This package provides a unified launch file (`main.launch`) that orchestrates all components of the door navigation project:

- **Simulation/Robot**: Gazebo simulation or real robot bringup
- **SLAM**: Mapping and localization using tiago_slam
- **Door Perception**: AprilTag-based door detection and modeling
- **Navigation**: Move_base control for autonomous navigation
- **Behavior Tree**: High-level door approach coordination
- **VICON**: Optional motion capture ground truth

## Launch Arguments

### Global Configuration

- `simulation` (default: `true`) - Use Gazebo simulation vs real robot
- `use_slam` (default: `true`) - Enable SLAM system
- `use_door_perception` (default: `true`) - Enable AprilTag door detection
- `use_navigation` (default: `true`) - Enable navigation control
- `use_behavior_tree` (default: `true`) - Enable BT-based door approach
- `use_vicon` (default: `false`) - Enable VICON motion capture
- `rviz` (default: `true`) - Launch RViz visualization

### Simulation Settings

- `world_file` (default: TIAGo door world) - Gazebo world file path
- `robot` (default: `tiago++`) - Robot model: `tiago++` (dual-arm), `titanium`, `steel`, or `iron`
- `gui` (default: `true`) - Show Gazebo GUI
- `gzpose` (default: origin) - Initial robot spawn pose

### Door Perception

- `door_config` (default: two-sided door config) - AprilTag configuration YAML

### Navigation

- `approach_cfg` (default: door approach config) - Door approach parameters

## Component Packages

The bringup integrates these packages:

- `tiago_door_sim` - Gazebo simulation environment
- `tiago_slam` - SLAM implementations
- `tiago_door_localization` - AprilTag-based door perception and doorway pose estimation
- `tiago_move_base_control` - Navigation goal sender and door approach goal generator
- `tiago_door_bt` - Behavior tree executor for door-opening pipeline
- `tiago_door_planning` - Action server for base/arm door-opening trajectory planning
- `tiago_arm_manipulation` - MoveIt-based arm motion and gripper control
- `tiago_vicon` - Motion capture integration and position logging

## Usage Examples

### Running by the main bringup

```bash
# default selection
roslaunch tiago_door_bringup main.launch
```

### Manually teleop robot to see door detection

Teleoperation without the controller by keyboard.

```bash
rosrun key_teleop key_teleop.py /key_vel:=/mobile_base_controller/cmd_vel
```

### Cleanup of a old map

```bash
rm ~/.ros/rtabmap.db
```

## RViz Configuration

A default RViz configuration is available at `rviz/tiago_door_nav.rviz`.

```bash
rviz -d $(rospack find tiago_door_bringup)/rviz/tiago_door_main.rviz
```
