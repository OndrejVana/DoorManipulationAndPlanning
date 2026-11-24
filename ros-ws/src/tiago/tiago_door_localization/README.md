# tiago_door_localization

ROS 1 (Melodic) package for AprilTag-based door perception. Detects the door's AprilTags and publishes the hinge
and handle poses in the map frame, with the tag's semantics (push/pull, hinge side). The door mission
(`tiago_door_mission`) averages these into a locked door model.

Topics are published only while a tag is in view; nothing is latched.

## Package structure

```text
tiago_door_localization/
├── config/
│   └── apriltag_two_sided_door.yaml   # Tag IDs, semantics and per-tag door models
├── launch/
│   └── door_preception_tags.launch    # Detector + both nodes
└── scripts/
    ├── door_tag_pose_node.py          # AprilTag detection → tag pose (camera, base, map), tag semantics
    └── door_model_from_tag_node.py    # Tag pose → hinge/handle/plane poses
```

## Launch

```bash
roslaunch tiago_door_localization door_preception_tags.launch
# bag replay: raw images under the recorded prefix
roslaunch tiago_door_localization door_preception_tags.launch image_transport:=raw \
  image_topic:=/slam_relay/xtion/rgb/image_rect_color camera_info_topic:=/slam_relay/xtion/rgb/camera_info
```

The detector's tag size is 0.144 m: the black square of the 0.18 m printed tags.

## Nodes

### `door_tag_pose_node`

- **Subscribes:** `/tag_detections` (`apriltag_ros/AprilTagDetectionArray`).
- **Picks a tag:** of the configured IDs in view, it keeps the current tag while visible.
- **Rejects** detections older than `max_age_s`.
- **Transforms** the pose to `base_footprint` and `map` at the image's timestamp.
- **Averages** the last 12 poses. The window restarts whenever the tag changes.

| Published topic | Type | Description |
| --- | --- | --- |
| `/door/tag_pose_map` | `geometry_msgs/PoseStamped` | Averaged tag pose in map |
| `/door/tag_pose_base`, `/door/tag_pose_camera` | `geometry_msgs/PoseStamped` | Tag pose in base and camera frames (debug) |
| `/door/tag_id` | `std_msgs/Int32` | Tag in use |
| `/door/door_side`, `/door/hinge_side`, `/door/interaction` | `std_msgs/String` | That tag's semantics from the config |

### `door_model_from_tag_node`

Applies the per-tag model (`models` in the config) to `/door/tag_pose_map`.

| Published topic | Type | Description |
| --- | --- | --- |
| `/door/handle_pose_map` | `geometry_msgs/PoseStamped` | Handle; z axis = door normal out of the tag's face |
| `/door/hinge_pose_map` | `geometry_msgs/PoseStamped` | Hinge; same header stamp as the handle of the same detection |
| `/door/plane_map` | `geometry_msgs/PoseStamped` | Tag position, z axis = door normal |
| `/door/hinge_axis_map` | `geometry_msgs/Vector3Stamped` | Hinge axis |

## Configuration (`config/apriltag_two_sided_door.yaml`)

| Parameter | Description |
| --- | --- |
| `apriltag/tag_ids` | `[0, 1, 2, 3]`: 0 push/hinge right, 1 push/hinge left, 2 pull/hinge left, 3 pull/hinge right |
| `apriltag/side_by_id`, `hinge_side_by_id`, `interaction_by_id` | Semantics per tag |
| `apriltag/preferred_side` | Prefer this side when several tags are visible (`""`: keep the current one) |
| `models/<id>` | Tag → handle and tag → hinge offsets per tag. Tag 1 was measured on the lab door; 0, 2 and 3 are guessed |
