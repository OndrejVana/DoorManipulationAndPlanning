# tiago_door_sim

ROS 1 (Melodic) package providing a Gazebo simulation environment for TIAGo door-opening experiments. Includes a small office world with a physics-based hinged door, AprilTag marker models, and pre-configured TIAGo spawn settings.

## Package structure

```text
tiago_door_sim/
├── config/
│   ├── ft_corrected_sim.yaml  # Wrist F/T adapter: hand mass * CoM, torque bias, lever arms
│   └── mission_start_poses.yaml  # Robot start poses with the right door in view, push and pull side
├── launch/
│   ├── sim_with_door.launch   # Main simulation launch file
│   └── door_edit.launch       # Helper launch for editing the door world
├── models/                    # Custom Gazebo models (door, AprilTag plates)
├── scripts/
│   ├── ft_corrected_sim.py    # Publishes /wrist_{side}_ft/corrected from the Gazebo wrist torque
│   └── mission_batch.sh       # Door mission once per start pose, results to ~/.ros/mission_batch/<side>.csv
└── worlds/
    └── small_office_with_door.world   # Gazebo world
```

## Dependencies

`gazebo_ros`, `roscpp`, `rospy`, `geometry_msgs`, `tf2_ros`

## Launch

```bash
# Default launch (Gazebo GUI, titanium robot)
roslaunch tiago_door_sim sim_with_door.launch

# Headless (no GUI)
roslaunch tiago_door_sim sim_with_door.launch gui:=false

# Different robot model
roslaunch tiago_door_sim sim_with_door.launch robot:=steel
```

The launch also relays `/scan_raw` to `/scan`: on the robot PAL's laser filters make `/scan`, which SLAM and
move_base read.

## Door mission batches

`rosrun tiago_door_sim mission_batch.sh push [pose names]` (or `pull`) starts a fresh headless sim at each pose of
`config/mission_start_poses.yaml`. It waits for play_motion (up with the arm controllers; a sim not up within
`STARTUP_TIMEOUT_S`, 900 s, is logged as such) and the start-up tuck (its retries would preempt the mission's tuck),
then `SETTLE_S` (60 s) for SLAM, then calls
`/door_mission/start` and waits for `/door_mission/result`. Each result goes to the CSV with the time taken, the true
hinge angle and the robot's final position. Doors: right door hinge at (0.6, -3); front (y > -3) tag 0 push, back
tag 2 pull.

## Launch arguments

| Argument | Default | Description |
| --- | --- | --- |
| `world_file` | `worlds/small_office_with_door.world` | Gazebo world file |
| `gui` | `true` | Show Gazebo GUI |
| `paused` | `false` | Start simulation paused |
| `robot` | `titanium` | Robot model: `iron`, `steel`, or `titanium` |
| `end_effector` | auto from robot | End-effector type (e.g. `pal-hey5`, `pal-gripper`) |
| `laser_model` | `sick-571` | Laser sensor model |
| `camera_model` | `orbbec-astra` | RGB-D camera model |
| `gzpose` | `0 0 0 0 0 0` | Initial robot spawn pose (`-x -y -z -R -P -Y`) |
| `tuck_arm` | `true` | Tuck robot arm at startup |
| `public_sim` | `false` | Use PAL public simulation stack |

## Including in another launch file

```xml
<include file="$(find tiago_door_sim)/launch/sim_with_door.launch">
  <arg name="world_file" value="$(find tiago_door_sim)/worlds/small_office_with_door.world"/>
  <arg name="gui" value="false"/>
  <arg name="robot" value="titanium"/>
</include>
```

## Notes

- The package sets `GAZEBO_MODEL_PATH` to include its own `models/` directory, so custom door and AprilTag models are found automatically.
- `use_sim_time` is set to `true` by the launch file.
- The door is a physics-based hinged joint — apply force via Gazebo or a ROS plugin to open it.
