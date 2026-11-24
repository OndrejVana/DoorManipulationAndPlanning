cmake_minimum_required(VERSION 3.0.2)
project(tiago_door_mission)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy
)

catkin_install_python(PROGRAMS
  scripts/door_mission_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
