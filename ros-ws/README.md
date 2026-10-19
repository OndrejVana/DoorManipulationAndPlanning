# TIAGo Development Workspace (ROS Melodic, Python 3.6)

Docker dev environment for the TIAGo door-opening packages. ROS Melodic with the project nodes on Python 3.6
(tf2, tf, PyKDL, eigenpy and moveit_commander are rebuilt for Python 3 inside the image).

## Prerequisites

- Docker with the compose plugin
- ~15 GB disk, 16 GB RAM (first image build compiles the PAL stack and MoveIt bindings, ~25 min)
- Compile parallelism is capped (`CATKIN_PKGS=2`, `CATKIN_JOBS=3`) to keep peak RAM ~8 GB; override with
  e.g. `CATKIN_JOBS=6 make up` / `make build CATKIN_JOBS=6` on bigger machines

## Usage

```bash
cd ros-ws/docker

make up         # build image (first time) and start the container
make shell      # open a shell in the container (allows X11 for Gazebo/RViz)
make build      # catkin build the project packages
make test       # run pytest for tiago_door_planning
make down       # stop the container
make logs       # follow container logs
```

`ros-ws/` is mounted at `/workspace/ros-ws`, so edits on the host are live and `build/`, `devel/` persist.
Inside the shell:

```bash
roslaunch tiago_door_bringup main.launch
```

For a real robot export `ROS_MASTER_URI` (and `ROS_IP`) on the host before `make up`; they are passed through.
Sync clocks first: [ROBOT_TIME_SYNC.md](../ROBOT_TIME_SYNC.md).

## SSH

```bash
ssh root@tiago-114c        # Ethernet
ssh root@192.168.130.114   # WiFi
```

## Resources

- [TIAGo tutorials](http://wiki.ros.org/Robots/TIAGo/Tutorials)
- [pal-robotics/tiago_tutorials](https://github.com/pal-robotics/tiago_tutorials)
