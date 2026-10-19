---
name: systematic-debugging
description: >-
  Investigate bugs, failing tests or builds, runtime errors (TF, MoveIt, move_base, BT) through reproduction,
  evidence and hypothesis testing. Use before speculative fixes.
---

# Systematic debugging

1. State expected vs observed. Reproduce with the narrowest command: a pytest, a single node via `rosrun`, a
   recorded bag (`ros-ws/src/tiago/record_bag.sh`, `tiago_door_bringup/launch/replay.launch`) or the sim.
2. Gather evidence: node logs (`~/.ros/log`), `rostopic echo/hz`, `rosrun tf tf_echo`, `rosnode info`,
   planner logs (`tiago_door_planning/planner_logs.py`), offline visualisers in `tiago_door_planning/scripts/`.
3. Narrow the failing layer: perception (tag → hinge/handle poses) → navigation → planning (lattice search,
   reachability, IK) → execution (base PID, arm trajectory). Separate code defects from environment causes:
   clock skew robot/laptop, `ROS_IP`/`ROS_MASTER_URI`, py2 vs py3 module shadowing in the overlay chain.
4. Form one concrete hypothesis with a prediction that could disprove it; test the cheapest probe first.
5. Discard disproved hypotheses and remove their temporary instrumentation before the next one.
6. Apply the smallest fix for the supported cause; add a regression test when a pure-logic boundary exists.
7. Rerun the original reproduction; finish with [verification](../verification-before-completion/SKILL.md).
