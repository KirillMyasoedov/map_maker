cmake_minimum_required(VERSION 3.0.2)
project(map_maker)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
)

catkin_python_setup()

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    src
)

install(PROGRAMS
        map_making_node.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

install(DIRECTORY
        launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )