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

### Common commands

```bash
rosrun key_teleop key_teleop.py /key_vel:=/mobile_base_controller/cmd_vel  # drive around to map
rosservice call /rtabmap/reset                                              # reset map (keeps database)
rm ~/.ros/rtabmap.db                                                        # start fresh
rosrun map_server map_saver -f ~/maps/my_map                                # save the 2D map
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
