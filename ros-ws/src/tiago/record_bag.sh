#!/usr/bin/env bash
#
# Usage : ./record_bag.sh [label]
#   
# Writes bags/<label>_<timestamp>/ with the bag (split every 2 GB) and next to it the parameter server dump, the
# URDF, the topic/node lists and which of the recorded topics were not being published.
#

set -u

LABEL="${1:-robot}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/bags/${LABEL}_$(date +%Y-%m-%d_%H-%M-%S)"

TOPICS=(
  # Frames and joints
  /tf /tf_static
  /joint_states
  # Base
  /mobile_base_controller/odom /mobile_base_controller/cmd_vel
  /base_imu
  /sonar_base
  # Laser: raw (door sensing in tiago_door_planning) and PAL's filtered one
  /scan_raw /scan
  # Wrist force/torque, only if the robot has the sensors
  /wrist_left_ft /wrist_right_ft
  /wrist_left_ft/corrected /wrist_right_ft/corrected
)
# Commands sent to the controllers and what they report back
for side in left right; do
  for group in arm hand; do
    controller="/${group}_${side}_controller"
    TOPICS+=(
      "$controller/command" "$controller/state"
      "$controller/follow_joint_trajectory/goal" "$controller/follow_joint_trajectory/result"
    )
  done
done
# Head camera (SLAM, AprilTags)
CAMERA=(
  /xtion/rgb/image_rect_color /xtion/rgb/camera_info
  /xtion/depth_registered/image_raw /xtion/depth_registered/camera_info
)
RELAY_PREFIX="/slam_relay"
TOPICS+=(
  /torso_controller/state /torso_controller/follow_joint_trajectory/goal
  /head_controller/state /head_controller/follow_joint_trajectory/goal
)

if ! rostopic list > /dev/null 2>&1; then
  echo "[record_bag] No ROS master at ${ROS_MASTER_URI:-<unset>}" >&2
  exit 1
fi
if [ -z "${ROS_IP:-}" ]; then
  echo "[record_bag] WARNING: ROS_IP is not set; the robot may not reach this machine" >&2
fi

if ! timeout 5 rostopic echo -n 1 /joint_states/header > /dev/null 2>&1; then
  echo "[record_bag] No message on /joint_states within 5 s. Check that the robot's hostname resolves to the" >&2
  echo "  robot's address on this network (getent hosts tiago-114c) and that a firewall does not block it." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
rostopic list | sort > "$OUT_DIR/topics.txt"
camera_prefix=""
if grep -qx "$RELAY_PREFIX${CAMERA[0]}" "$OUT_DIR/topics.txt"; then
  camera_prefix="$RELAY_PREFIX"
  echo "[record_bag] Live SLAM relay found: recording the camera from $RELAY_PREFIX"
fi
TOPICS+=("${CAMERA[@]/#/$camera_prefix}")
rosnode list > "$OUT_DIR/nodes.txt" 2>&1
rosparam dump "$OUT_DIR/params.yaml"
rosparam get /robot_description > "$OUT_DIR/robot_description.urdf" 2>&1
{
  echo "label: $LABEL"
  echo "start: $(date --iso-8601=seconds)"
  echo "ros_master_uri: ${ROS_MASTER_URI:-}"
  echo "ros_ip: ${ROS_IP:-}"
  if [ -n "$camera_prefix" ]; then
    echo "camera_prefix: $camera_prefix"
    echo "replay_remaps: $(for topic in "${CAMERA[@]}"; do printf '%s:=%s ' "$camera_prefix$topic" "$topic"; done)"
  fi
} > "$OUT_DIR/info.txt"

printf '%s\n' "${TOPICS[@]}" | sort | comm -23 - "$OUT_DIR/topics.txt" > "$OUT_DIR/missing_topics.txt"
if [ -s "$OUT_DIR/missing_topics.txt" ]; then
  echo "[record_bag] Not published right now (recorded if they appear):"
  sed 's/^/  /' "$OUT_DIR/missing_topics.txt"
fi

echo "[record_bag] Recording to: $OUT_DIR"
echo "[record_bag] Press Ctrl+C to stop."
# Ctrl+C stops rosbag; the script itself carries on to the summary.
trap 'true' INT
rosbag record --split --size=2048 --buffsize=2048 -O "$OUT_DIR/$LABEL" "${TOPICS[@]}"
trap - INT

echo "end: $(date --iso-8601=seconds)" >> "$OUT_DIR/info.txt"
# The recorder may still be closing its .bag.active files after rosbag itself returned.
for _ in $(seq 30); do
  compgen -G "$OUT_DIR/*.active" > /dev/null || break
  sleep 1
done
rosbag info "$OUT_DIR"/*.bag > "$OUT_DIR/bag_info.txt" 2>&1
if ! grep -q "^messages:" "$OUT_DIR/bag_info.txt"; then
  echo "[record_bag] WARNING: the bag holds no messages, see $OUT_DIR/bag_info.txt" >&2
fi
# The container runs as root; hand the recording to the owner of this repo.
chown -R --reference="$SCRIPT_DIR" "$OUT_DIR" 2>/dev/null
echo "[record_bag] Saved: $OUT_DIR ($(du -sh "$OUT_DIR" | cut -f1))"
