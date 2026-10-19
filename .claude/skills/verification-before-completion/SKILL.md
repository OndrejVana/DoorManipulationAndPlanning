---
name: verification-before-completion
description: >-
  Establish evidence before claiming a change is done, fixed or passing. Use at completion and after material
  fixes; scale to the change.
---

# Verification before completion

Claim completion only to the extent checks of the final state support it.

1. Map each acceptance criterion to an observable check. Run the narrowest one first.
2. Workspace checks (from `ros-ws/docker`, container up via `make up`):
   - `make build` — catkin build of project packages (needed after msg/srv/action, CMake or C++ changes).
   - `make test` — pytest for `tiago_door_planning/tests`.
   - `docker compose exec dev bash -lc "python3 -m py_compile <files>"` — py3.6 syntax for changed files.
   - Import check for nodes: `docker compose exec dev bash -lc "python3 -c 'import <module>'"`.
   - Host: `ruff check <files>` with the root `ruff.toml`.
3. Launch-level checks (`roslaunch tiago_door_bringup main.launch gui:=false`) only when the change affects
   runtime wiring; report what was observed (nodes up, errors in log), not assumed.
4. Inspect the final diff and run `git diff --check`. Don't credit pre-existing changes to the task.
5. Report commands, results and what stayed unverified (e.g. real robot behaviour). Never drive the real robot
   without the user asking.
