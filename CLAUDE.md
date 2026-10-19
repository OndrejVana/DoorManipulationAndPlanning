# Workspace instructions

TIAGo++ autonomous door opening. ROS 1 Melodic catkin workspace; project nodes run on **Python 3.6**.

## Ownership

- `ros-ws/docker/`: dev image (Dockerfile, compose, entrypoint, Makefile), PAL `.rosinstall`.
- `ros-ws/src/tiago/<package>/`: project catkin packages; [package overview](ros-ws/src/tiago/README.md).
  - `tiago_door_bringup`: `main.launch` entry point; component flags (`use_slam`, `use_navigation`, ...).
  - `tiago_door_planning`: planning/execution action servers; library in `src/tiago_door_planning/`, tests in
    `tests/`, offline reachability maps in `maps/*.npz`, tuning/visualisation scripts in `scripts/`.
  - `tiago_door_bt` (py_trees), `tiago_door_localization` (AprilTag), `tiago_arm_manipulation` (MoveIt),
    `tiago_move_base_control` (move_base + C++ `DoorMaskLayer` costmap plugin), `tiago_door_sim` (Gazebo),
    `tiago_slam`, `tiago_vicon`.
- `ros-ws/scripts/`: standalone helpers (AprilTag generation). `docs/`: thesis figures (LaTeX/TikZ).
- `ROBOT_TIME_SYNC.md`: chrony setup between robot and laptop. Everything runs on the laptop.

## Environment

- Everything builds and runs in Docker: `cd ros-ws/docker && make up | shell | build | test | down`.
  Repo `ros-ws/` is bind-mounted at `/workspace/ros-ws`; `build/`, `devel/` persist on the host.
- Overlay chain: `/opt/ros/melodic` (py2.7) → `/tiago_public_ws` (PAL stack, py2.7) → `/py3_ws`
  (tf2_py, tf, PyKDL, eigenpy, moveit_commander rebuilt for py3) → `/workspace/ros-ws/devel` (built with
  `-DPYTHON_EXECUTABLE=/usr/bin/python3`).
- Python deps without a py3 rebuild in `/py3_ws` are unavailable to project nodes (e.g. `cv_bridge`, PyKDL-based
  `tf_conversions`). Adding one means adding it to the `/py3_ws` stage of the Dockerfile.
- `PYTHONPATH` holds py3 (`/py3_ws`) and py2.7 paths together; a py2 ROS tool importing a rebuilt C extension
  (`tf2_py`, `PyKDL`) can fail. `/etc/python2.7/sitecustomize.py` strips `/python3/` paths for python2; py2
  `moveit_commander` still picks up the py3 `libeigenpy` (`_ARRAY_API` warning) — only py3 nodes use MoveIt.
- Builds are memory-capped (`make build CATKIN_PKGS=2 CATKIN_JOBS=3` defaults); don't raise them on the 16 GB
  dev machine — uncapped catkin crashed it.

## Conventions

- Python 3.6 only: shebang `#!/usr/bin/env python3`; no `from __future__`, dataclasses, walrus, `f"{x=}"`,
  `from __future__ import annotations`, `subprocess.run(capture_output=...)`.
- Keep pure planning logic in `tiago_door_planning/src/` free of rospy where possible so it stays unit-testable.
- Python: root `ruff.toml` (run ruff on the host; it does not install on 3.6). Markdown/YAML/XML: `.editorconfig`.
- Changing `.msg/.srv/.action` or `CMakeLists.txt` requires `make build`; Python edits under `scripts/` do not.

## Efficient agent work

- Start searches in the affected package; skip `build/`, `devel/`, `logs/`, `*.npz`, `docs/figs/*.pdf`.
- Verify inside the container (`make build`, `make test`); robot runs are the user's call.
