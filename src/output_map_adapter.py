import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from point_cloud_classifier.msg import PointsList
from std_msgs.msg import Header
from .messages_creator import create_occupancy_grid_message, create_point_cloud_message, create_labeled_points_message
import numpy as np
import os
from time import time


class OutputMapAdapter(object):
    def __init__(self,
                 non_passable_grid_topic="/non_passable_grid",
                 passable_grid_topic="/passable_grid",
                 point_cloud_topic="/point_cloud_common",
                 saved_points_topic="/saved_points",
                 points_topic="/points_common",
                 labels_topic="/labels_common"):
        self.__non_passable_grid_publisher = rospy.Publisher(non_passable_grid_topic, OccupancyGrid, queue_size=10)
        self.__passable_grid_publisher = rospy.Publisher(passable_grid_topic, OccupancyGrid, queue_size=10)
        self.__point_cloud_publisher = rospy.Publisher(point_cloud_topic, PointCloud2, queue_size=10)
        self.__saved_points_publisher = rospy.Publisher(saved_points_topic, PointCloud2, queue_size=10)
        self.__points_publisher = rospy.Publisher(points_topic, PointsList, queue_size=10)
        self.__labels_publisher = rospy.Publisher(labels_topic, PointsList, queue_size=10)

        self.product_path = "/home/kirill/catkin_ws/src/map_maker/product"

    def publish_map(self,
                    non_passable_grid,
                    passable_grid,
                    height,
                    width,
                    resolution,
                    origin_pose,
                    point_cloud=None,
                    labels=None,
                    visualize_point_cloud=False):
        # Creating messages
        header = Header()
        non_passable_grid_message = create_occupancy_grid_message(header,
                                                                  non_passable_grid,
                                                                  height,
                                                                  width,
                                                                  resolution,
                                                                  origin_pose)
        passable_grid_message = create_occupancy_grid_message(header,
                                                              passable_grid,
                                                              height,
                                                              width,
                                                              resolution,
                                                              origin_pose)

        # Publish messages
        self.__non_passable_grid_publisher.publish(non_passable_grid_message)
        self.__passable_grid_publisher.publish(passable_grid_message)
        if os.path.exists(os.path.join(self.product_path, "labeled_points.npz")):
            common_points = np.load(os.path.join(self.product_path, "labeled_points.npz"))["arr_0"]
            # saved_points = common_points[common_points[:, 3] // 1000 == 2][:, :3]
            # saved_labels = common_points[common_points[:, 3] // 1000 == 2][:, 3]
            saved_points = common_points[:, :3]
            saved_labels = common_points[:, 3]

            point_cloud_message = create_point_cloud_message(saved_points, saved_labels, header)
            self.__saved_points_publisher.publish(point_cloud_message)

        # If visualize point cloud, create the message and publish
        if visualize_point_cloud:
            point_cloud_message = create_point_cloud_message(point_cloud, labels, header)
            self.__point_cloud_publisher.publish(point_cloud_message)

