#!/usr/bin/env bash
set -e

# Chain: /opt/ros/melodic -> /tiago_public_ws -> /py3_ws -> /workspace/ros-ws (when built).
# Sourced here so `docker compose exec/run <cmd>` sees it, not just interactive shells.
source /etc/profile.d/10-tiago-ros.sh

exec "$@"
