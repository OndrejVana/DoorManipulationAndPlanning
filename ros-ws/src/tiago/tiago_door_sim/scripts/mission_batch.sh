#!/usr/bin/env bash
# Runs the door mission in Gazebo once per start pose: mission_batch.sh <push|pull> [pose names...] (default: all).
# For each pose: fresh headless sim + bringup, SLAM settles, /door_mission/start, wait for /door_mission/result,
# then log the outcome, the true hinge angle and where the robot ended. Results: ~/.ros/mission_batch/<side>.csv

set -uo pipefail
side=$1
shift
poses_file=$(rospack find tiago_door_sim)/config/mission_start_poses.yaml
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  mapfile -t names < <(python3 -c "import yaml,sys; print('\n'.join(yaml.safe_load(open('$poses_file'))['$side']))")
fi
out=${MISSION_BATCH_DIR:-$HOME/.ros/mission_batch}
mkdir -p "$out"
csv=$out/$side.csv
[ -f "$csv" ] || echo "pose,x,y,yaw,result,seconds,hinge_deg,robot_x,robot_y" > "$csv"
settle=${SETTLE_S:-60}
timeout=${MISSION_TIMEOUT_S:-1200}
startup_timeout=${STARTUP_TIMEOUT_S:-900}

stop_sim() {
  pkill -INT -f "bin/roslaunch" 2>/dev/null
  for _ in $(seq 30); do pgrep -f "gzserver|bin/roslaunch" > /dev/null || break; sleep 1; done
  pkill -9 -f "gzserver|bin/roslaunch|rosmaster|rosout|__name:=" 2>/dev/null
  while rostopic list > /dev/null 2>&1; do sleep 1; done
}

for name in "${names[@]}"; do
  stop_sim
  read -r x y yaw < <(python3 -c "import yaml; p=yaml.safe_load(open('$poses_file'))['$side']['$name']; print(*p)")
  log=$out/${side}_${name}_$(date +%H%M%S).log
  echo "== $side/$name ($x, $y, $yaw) -> $log"
  roslaunch tiago_door_bringup main.launch simulation:=true gui:=false \
    gzpose:="-x $x -y $y -z 0 -R 0 -P 0 -Y $yaw" > "$log" 2>&1 &
  launch=$!

  up_by=$(( $(date +%s) + startup_timeout ))
  until timeout 5 rostopic echo -n 1 /play_motion/status > /dev/null 2>&1; do
    [ "$(date +%s)" -lt "$up_by" ] || break
    sleep 5
  done

  while rosnode list 2>/dev/null | grep -q "^/tuck_arm$"; do
    [ "$(date +%s)" -lt "$up_by" ] || break
    sleep 5
  done
  if [ "$(date +%s)" -ge "$up_by" ]; then
    echo "$name,$x,$y,$yaw,\"sim did not come up\",0,nan,nan,nan" | tee -a "$csv"
    stop_sim
    wait $launch
    continue
  fi
  sleep "$settle"
  start=$(date +%s)
  rosservice call --wait /door_mission/start > /dev/null
  result=$(timeout "$timeout" rostopic echo -n 1 /door_mission/result 2>/dev/null | sed -n 's/^data: //p' | tr -d '"')
  result=${result:-timeout}
  seconds=$(( $(date +%s) - start ))
  hinge_deg=$(rosservice call /gazebo/get_joint_properties "joint_name: 'hinged_door_right::hinge'" 2>/dev/null \
    | python3 -c "import sys,yaml,math; d=yaml.safe_load(sys.stdin.read()); print(round(abs(math.degrees(d['position'][0])), 1))" \
    2>/dev/null || echo nan)
  read -r rx ry < <(rosservice call /gazebo/get_model_state "model_name: 'tiago_dual'" 2>/dev/null \
    | python3 -c "import sys,yaml; d=yaml.safe_load(sys.stdin.read()); p=d['pose']['position']; print(round(p['x'],2), round(p['y'],2))" \
    2>/dev/null || echo "nan nan")
  echo "$name,$x,$y,$yaw,\"$result\",$seconds,$hinge_deg,$rx,$ry" | tee -a "$csv"
  stop_sim
  wait $launch
done
