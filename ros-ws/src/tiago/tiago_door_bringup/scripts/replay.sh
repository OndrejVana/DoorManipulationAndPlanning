#!/usr/bin/env bash
# Replays a record_bag.sh recording directory: replay.sh <bag dir> [roslaunch args...]
set -euo pipefail
dir=$(cd "$1" && pwd)
shift
bags=$(ls "$dir"/*.bag | tr '\n' ' ')
prefix=$(sed -n 's/^camera_prefix: //p' "$dir/info.txt" 2>/dev/null || true)
urdf=$(mktemp --suffix=.urdf)
python3 -c 'import sys, yaml; text = open(sys.argv[1]).read(); sys.stdout.write(text if text.lstrip().startswith("<") else yaml.safe_load(text))' \
  "$dir/robot_description.urdf" > "$urdf"
exec roslaunch tiago_door_bringup replay.launch bags:="$bags" urdf:="$urdf" \
  camera_prefix:="${prefix}" "$@"
