# TIAGo SLAM

RTAB-Map based SLAM for the TIAGo robot using RGBD odometry, IMU, and a virtual laser scan derived from the depth camera.

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

### Old mapping nodes still running

```bash
rosnode kill /amcl /map_server /map_aligner_node /map_configuration_server /compressed_map_publisher
```

### Robot not moving with teleop

```bash
rosnode kill /twist_mux
```

### Clock sync issues

```bash
# Sync laptop clock to robot
sudo ntpdate -u tiago-114c

# Check chrony status
chronyc tracking
```
