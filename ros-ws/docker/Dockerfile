FROM osrf/ros:melodic-desktop-full-bionic

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        nano \
        vim \
        tree \
        htop \
        gdb \
        psmisc \
        iputils-ping \
        ssh \
        terminator \
        dbus-x11 \
        python-catkin-tools \
        python-wstool \
        python3-vcstool \
        python-scipy \
        python-networkx \
        python-pip \
        ros-melodic-robot-localization \
        ros-melodic-rtabmap-ros \
        ros-melodic-depthimage-to-laserscan \
        ros-melodic-vrpn-client-ros \
        ros-melodic-apriltag-ros \
        ros-melodic-tf2-geometry-msgs \
        ros-melodic-image-transport \
        ros-melodic-cv-bridge \
        ros-melodic-eigenpy \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        mesa-utils \
        libqt5opengl5 \
        qt5-default \
    && rm -rf /var/lib/apt/lists/*

# Python 3.6 (bionic's python3) runtime for the project nodes. ROS Melodic itself stays on Python 2.7.
# py_trees 0.6.9 (apt) is pure Python and py3-compatible, so python3 imports it from the melodic path.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-yaml \
        python3-pil \
        python3-pytest \
        python3-empy \
        python3-rospkg-modules \
        python3-catkin-pkg-modules \
        python3-pyassimp \
        python3-sip-dev \
        python3-pydot \
        libboost-python-dev \
        ros-melodic-py-trees \
    && rm -rf /var/lib/apt/lists/*

# Compile parallelism: packages in parallel x make jobs per package. Defaults keep peak RAM ~8 GB;
# uncapped catkin (nproc x nproc) exhausts a 16 GB host on the MoveIt sources.
ARG CATKIN_PKGS=2
ARG CATKIN_JOBS=3

# PAL TIAGo public stack (Python 2, as released by PAL).
ARG PAL_WS=/tiago_public_ws
ARG ROSDEP_IGNORE="opencv2 opencv2-nonfree pal_laser_filters speed_limit_node sensor_to_cloud hokuyo_node libdw-dev python-graphitesend-pip python-statsd pal_filters pal_vo_server pal_usb_utils pal_pcl pal_pcl_points_throttle_and_filter pal_karto pal_local_joint_control camera_calibration_files pal_startup_msgs pal-orbbec-openni2 dummy_actuators_manager pal_local_planner gravity_compensation_controller current_limit_controller dynamic_footprint dynamixel_cpp tf_lookup opencv3 omni_drive_controller pal_moveit_plugins pal_moveit_capabilities pal_loc_measure pal_map_manager ydlidar_ros_driver pal_motion_loader pal_navigation_msgs tiago_dual_description_calibration joint_impedance_trajectory_controller tiago_dual_2dnav"
COPY tiago_public-melodic.rosinstall ${PAL_WS}/tiago_public-melodic.rosinstall
RUN mkdir -p ${PAL_WS}/src && cd ${PAL_WS} \
    && vcs import src < tiago_public-melodic.rosinstall \
    && apt-get update \
    && rosdep install --from-paths src --ignore-src -y --rosdistro melodic --skip-keys="${ROSDEP_IGNORE}" \
    && rm -rf /var/lib/apt/lists/* \
    && source /opt/ros/melodic/setup.bash \
    && catkin config --extend /opt/ros/melodic \
    && catkin build -p ${CATKIN_PKGS} -j ${CATKIN_JOBS} -DCATKIN_ENABLE_TESTING=0 \
    && catkin clean --build --yes

# Python 3 rebuild of the compiled ROS Python libraries the project imports, pinned to the final Melodic
# releases so they stay ABI-compatible with the apt packages they overlay:
# tf2_py/tf2_ros/tf2_geometry_msgs, tf, PyKDL (tf2_geometry_msgs needs it), eigenpy + moveit_commander.
ARG PY3_WS=/py3_ws
RUN mkdir -p ${PY3_WS}/src && cd ${PY3_WS}/src \
    && git clone --depth 1 -b 0.6.5 https://github.com/ros/geometry2.git \
    && git clone --depth 1 -b 1.12.1 https://github.com/ros/geometry.git \
    && git clone --depth 1 -b v1.4.0 https://github.com/orocos/orocos_kinematics_dynamics.git \
    && git clone --depth 1 --recursive -b v2.9.2 https://github.com/stack-of-tasks/eigenpy.git \
    && git clone --depth 1 -b 1.0.11 https://github.com/ros-planning/moveit.git \
    && for d in geometry2/*/ geometry/*/ orocos_kinematics_dynamics/*/ moveit/*/ moveit/moveit_ros/*/; do \
           touch "${d}CATKIN_IGNORE"; \
       done \
    && rm -f geometry2/tf2_py/CATKIN_IGNORE geometry2/tf2_ros/CATKIN_IGNORE geometry2/tf2_geometry_msgs/CATKIN_IGNORE \
        geometry/tf/CATKIN_IGNORE \
        orocos_kinematics_dynamics/python_orocos_kdl/CATKIN_IGNORE \
        moveit/moveit_ros/CATKIN_IGNORE moveit/moveit_ros/planning_interface/CATKIN_IGNORE \
        moveit/moveit_commander/CATKIN_IGNORE \
    # tf2_py hard-codes Python 2; Boost 1.65 names its Python 3 component "python3" (moveit handles >= 1.67 only).
    && sed -i 's/find_package(PythonLibs 2 REQUIRED)/find_package(PythonLibs 3 REQUIRED)/' \
        geometry2/tf2_py/CMakeLists.txt \
    && sed -i 's/set(BOOST_PYTHON_COMPONENT python)/set(BOOST_PYTHON_COMPONENT python3)/' \
        moveit/moveit_ros/planning_interface/CMakeLists.txt \
    && cd ${PY3_WS} \
    && source ${PAL_WS}/devel/setup.bash \
    && catkin config --extend ${PAL_WS}/devel --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DCATKIN_ENABLE_TESTING=0 \
        -DBUILD_TESTING=OFF \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON_VERSION=3 \
    && catkin build -p 1 -j ${CATKIN_JOBS} \
    && catkin clean --build --yes

RUN python3 -m pip install --no-cache-dir --upgrade "pip==21.3.1" \
    && python3 -m pip install --no-cache-dir pin==2.6.4

# Tk backend for the matplotlib visualisers in tiago_door_planning/scripts (matplotlib.use('TkAgg')).
RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3-tk \
    && rm -rf /var/lib/apt/lists/*

# No GPU passthrough: Gazebo and RViz render on Mesa's llvmpipe, which works on any host.
ENV ROS_PYTHON_VERSION=3 \
    QT_QPA_PLATFORM=xcb \
    LIBGL_ALWAYS_SOFTWARE=1 \
    XDG_RUNTIME_DIR=/tmp/runtime-ros \
    NO_AT_BRIDGE=1
RUN mkdir -p /tmp/runtime-ros && chmod 0700 /tmp/runtime-ros

# The project workspace is bind-mounted here by docker-compose; build/ and devel/ persist on the host.
WORKDIR /workspace/ros-ws

# `docker exec` bypasses the entrypoint, so shells must source ROS themselves. profile.d covers
# login shells (`bash -lc`, which does NOT read .bashrc); .bashrc covers interactive non-login ones.
RUN printf '%s\n' \
        '[ -n "$TIAGO_ENV_SOURCED" ] && return' \
        'export TIAGO_ENV_SOURCED=1' \
        'if [ -f /workspace/ros-ws/devel/setup.bash ]; then' \
        '  source /workspace/ros-ws/devel/setup.bash' \
        'else' \
        '  source /py3_ws/devel/setup.bash' \
        'fi' \
        '# py3 setup only exports lib/python3; the pure-Python Melodic libs (rospy, msgs, py_trees) live in' \
        '# python2.7 paths and work on py3, so append them after the rebuilt py3 modules.' \
        'export PYTHONPATH="$PYTHONPATH:/tiago_public_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages"' \
        > /etc/profile.d/10-tiago-ros.sh \
    && echo 'source /etc/profile.d/10-tiago-ros.sh' >> /root/.bashrc \
    && printf '%s\n' '' \
        '# Python 3 overlay paths share PYTHONPATH; keep python2 off their C extensions (tf2_py, PyKDL).' \
        'import sys' \
        'sys.path[:] = [p for p in sys.path if "/python3/" not in p]' \
        >> /etc/python2.7/sitecustomize.py \
    && sed -i 's/^mesg n.*/tty -s \&\& mesg n || true/' /root/.profile

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["sleep", "infinity"]
