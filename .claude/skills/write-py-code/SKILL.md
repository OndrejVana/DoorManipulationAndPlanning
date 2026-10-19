---
name: write-py-code
description: Write Python 3.6 ROS Melodic nodes, library code and tests in this workspace using local conventions.
---

# Python development

- Use the affected package as the source of truth for its API, parameters (`config/*.yaml`) and tests.
- Add short docstrings on classes and public functions; comment only non-obvious logic.

## Language

- Target **Python 3.6** (bionic `python3`, the only py3 the ROS overlay supports). Not available: dataclasses,
  `from __future__ import annotations`, walrus, `f"{x=}"`, `subprocess.run(capture_output=/text=)`,
  `math.prod/dist/isqrt`, `str.removeprefix`, `contextlib.nullcontext`. f-strings and type hints are fine.
- Executables: shebang `#!/usr/bin/env python3`, executable bit set, listed in `catkin_install_python()`.
- Only import compiled ROS Python modules that exist for py3 in `/py3_ws` (tf2_py, tf2_ros, tf2_geometry_msgs,
  tf, PyKDL, moveit_commander) or are pure Python. `cv_bridge` is not rebuilt.
- Match nearby style (`rospy.get_param("~...")` loading in `__init__`, `rospy.loginfo` with `%` formatting).
- Check changed files with the root `ruff.toml` on the host; avoid reformatting untouched code.

## Design

- Keep ROS I/O (params, pubs/subs, actions, TF) at the node boundary; pass plain values/messages into logic
  modules (pattern: `tiago_door_planning/src/tiago_door_planning/*` used by `scripts/door_planning_server.py`).
- Catch specific exceptions (`tf2_ros.LookupException`, ...) where they can be handled; log and fail the action.

## Testing

- pytest tests live in `<package>/tests/test_<module>.py` (see `tiago_door_planning/tests`). Use plain objects or
  stubs instead of a running ROS master. Run with `make test` from `ros-ws/docker`.
- Arrange / Act / Assert sections; one behaviour per test.
