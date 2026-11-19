# TIAGo SLAM

RTAB-Map based SLAM for the TIAGo robot: laser scan-to-map odometry on top of the wheel odometry, visual loop
closure detection from the head camera, and laser ICP on every graph link.

Used as the mapping and localization backend for door-opening tasks via `tiago_door_bringup`.

## Usage

### TIAGo SLAM (primary — used for door opening)

Launched automatically by `main.launch`. To run standalone:

```bash
roslaunch tiago_slam tiago_slam.launch
```

Publishes:

- `/map` — 2D occupancy grid
- `/slam_pose` — robot pose in map frame (`PoseStamped`)
- TF: `map → odom → base_footprint`

`odometry` arg:

- `icp` (default): `icp_odometry` on `/scan`, guessed from the wheel odometry. Corrects wheel slip and does not drift
  while standing. The SLAM runs on `/slam/tf` (`map → odom_icp → odom → base_footprint`); `slam_bridge` publishes
  the composed `map → odom` on `/tf` and `/slam_pose`.
- `ekf`: wheels + visual odometry + IMU fused by `robot_localization` (previous setup).
- `wheel`: wheel odometry from `/tf`.

### RTAB-Map SLAM (development / exploration)

- Simulation

```bash
roslaunch tiago_slam rtab_slam.launch world_name:="small_office"
```

- Real robot

```bash
roslaunch tiago_slam rtab_slam.launch simulation:=false
```

- Common commands

```bash
# Visualize
rviz -d $(rospack find tiago_slam)/rviz/rtab_slam.rviz

# Manual control
rosrun key_teleop key_teleop.py /key_vel:=/mobile_base_controller/cmd_vel

# Reset map (keeps database)
rosservice call /rtabmap/reset

# Delete database and start fresh
rm ~/.ros/rtabmap.db

# Save 2D map
rosrun map_server map_saver -f ~/maps/my_map
```

### RTAB-Map Mono SLAM

- Simulation

```bash
roslaunch tiago_slam rtab_mono_slam.launch world_name:="small_office"
```

- Real robot

```bash
roslaunch tiago_slam rtab_mono_slam.launch simulation:=false
```

- Common commands

```bash
rviz -d $(rospack find tiago_slam)/rviz/rtab_mono_slam.rviz
rosrun key_teleop key_teleop.py /key_vel:=/mobile_base_controller/cmd_vel
rosservice call /rtabmap/reset
rm ~/.ros/rtabmap.db
rosrun map_server map_saver -f ~/maps/my_map
```

## Troubleshooting

### PAL mapping nodes and automatic camera movement

```bash
rosrun tiago_slam kill_pal_nav_nodes.py
rosrun tiago_slam kill_head_nodes.py
```

### Robot not moving with teleop

```bash
rosnode kill /twist_mux
```
