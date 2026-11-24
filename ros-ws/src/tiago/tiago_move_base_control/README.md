# tiago_move_base_control

ROS 1 (Melodic) package with the `move_base` setup for TIAGo and the `DoorMaskLayer` costmap plugin.

- The door mission (`tiago_door_mission`) sends move_base goals through the action API itself.
- `tiago_door_planning` plans on this package's global costmap.

## Package structure

```text
tiago_move_base_control/
├── launch/
│   └── tiago_move_base.launch   # move_base (ns /tiago_move_base) with both costmaps
├── src/DoorMaskLayer.cpp        # costmap plugin
├── include/                     # plugin header
└── plugin.xml                   # plugin registration
```

## move_base

`roslaunch tiago_move_base_control tiago_move_base.launch` (started by `main.launch use_navigation:=true`).

- **Namespace:** `/tiago_move_base/move_base`.
- **Planners:** NavfnROS (global) and TrajectoryPlannerROS (local).
- **Remaps:** `cmd_vel` goes to `/mobile_base_controller/cmd_vel`. `execute_door_segment` publishes to the same
  topic, so move_base must be idle during the opening; the mission ensures this.
- **Base:** `robot_radius` 0.30, the same as `collision/base_radius` in `tiago_door_planning`.
- **Global costmap:** frame `map`, not rolling. Layers in order:
  1. `static_layer`: rtabmap's live `/map`.
  2. `obstacle_layer`: `/scan`, so obstacles not yet in the map count.
  3. `door_mask`.
  4. `inflation_layer`.

  `open_door` reads this costmap (`/tiago_move_base/move_base/global_costmap/costmap`) afresh for every plan.
- **Local costmap:** frame `odom`, rolling 4×4 m. Layers: obstacle (`/scan`) and inflation. It has no
  door mask: near the robot, the leaf is a real obstacle.

Only the base laser feeds the obstacle layers. The former `/depth_scan` (`depthimage_to_laserscan` in `tiago_slam`)
was stamped in `base_footprint` although its ranges are measured from the head camera, which put everything it saw
about 20 cm too close; with the head turned to the door its scan plane is not horizontal either. Obstacles above the
laser plane (table tops) are in no layer now: the icp SLAM grid is built from the laser too. A point cloud obstacle
layer would add them.

## DoorMaskLayer

Clears the cells inside a polygon while enabled. The mission publishes the door leaf's swing sector, from closed to
`mask/max_angle_deg` and grown by `mask/margin`. The planner models the leaf itself, so the leaf the SLAM map and the
sensors see must not block it twice. Without the mask, a closed leaf in the map blocks push-door base paths, and the
leaf positions rtabmap adds while the door opens block later attempts.

| Parameter (`~<costmap>/door_mask/`) | Default | Meaning |
| --- | --- | --- |
| `polygon_topic` | `/door_mask/polygon` | `geometry_msgs/PolygonStamped`, any frame TF can map into the costmap's |
| `enabled_topic` | `/door_mask/enabled` | `std_msgs/Bool` |
| `set_unknown_instead_of_free` | `false` | Write NO_INFORMATION instead of FREE_SPACE |

- The polygon is transformed into the costmap's frame on every update.
- Clearing costmaps (`reset`) keeps the polygon.
- Disabling the mask or moving the polygon repaints the previously masked area from the layers below.
