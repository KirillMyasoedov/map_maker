#!/usr/bin/env python3
import rospy
from src import MapMakingNodeFactory

if __name__ == "__main__":
    try:
        rospy.init_node("map_making_node")
        factory = MapMakingNodeFactory()
        factory.make_map_making_node()

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
