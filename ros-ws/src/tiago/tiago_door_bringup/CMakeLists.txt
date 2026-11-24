cmake_minimum_required(VERSION 3.0.2)
project(tiago_door_bringup)

find_package(catkin REQUIRED COMPONENTS
  roslaunch
  roscpp
  rospy
)

catkin_package()

# Check launch files for syntax errors
roslaunch_add_file_check(launch)

install(DIRECTORY launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS scripts/replay.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
