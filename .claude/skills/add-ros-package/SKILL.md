---
name: add-ros-package
description: Create a new ROS 1 Melodic catkin package under ros-ws/src/tiago following the workspace layout.
---

# Add a ROS package

Create `ros-ws/src/tiago/<tiago_*>/`; follow [Python conventions](../write-py-code/SKILL.md).

```text
<package>/
  CMakeLists.txt     catkin; catkin_install_python(PROGRAMS scripts/...) for every node
  package.xml        format 2; python deps as python3-* keys
  README.md          purpose, nodes, topics/services, launch args (short, like sibling packages)
  setup.py           only if it has an importable module: distutils + catkin_pkg generate_distutils_setup
  src/<package>/     importable library (with catkin_python_setup())
  scripts/           node entry points (#!/usr/bin/env python3, chmod +x)
  launch/ config/    launch files; parameter YAML loaded with <rosparam>
  msg/ srv/ action/  interfaces (add message_generation / message_runtime)
  tests/             pytest
```

- Copy structure from the closest sibling (`tiago_door_planning` for library + actions,
  `tiago_door_localization` for plain nodes, `tiago_move_base_control` for C++ plugins).
- Wire it into `tiago_door_bringup/launch/main.launch` behind a `use_<feature>` arg if it is part of the system.
- Add a row to `ros-ws/src/tiago/README.md` and the root `README.md` table.
- New apt/pip dependencies go in `ros-ws/docker/Dockerfile`; py3 compiled ROS deps go in its `/py3_ws` stage.
- Verify with `make build` (and `make test` if tests were added).
