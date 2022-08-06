import numpy as np
import ros_numpy
from point_cloud_classifier.msg import PointsList
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
import message_filters
import rospy
from .transformer import transform_pcl
from time import time


class InputMapAdapter(object):
    def __init__(self,
                 labeled_points_topic="/labeled_points_",
                 rtabmap_grid_topic="/rtabmap/grid_map",
                 synchronize=False):
        cameras_number_parameter_name = rospy.search_param("cameras_number")
        self.cameras_number = rospy.get_param(cameras_number_parameter_name)

        self.__subscribers = []
        if synchronize:
            grid_map_subscriber = message_filters.Subscriber(rtabmap_grid_topic, OccupancyGrid)
            self.__subscribers.append(grid_map_subscriber)
            for i in range(self.cameras_number):
                self.__subscribers.append(message_filters.Subscriber(labeled_points_topic + str(i + 1), PointsList))

            time_synchronizer = message_filters.ApproximateTimeSynchronizer(self.__subscribers,
                                                                            100,
                                                                            0.8)
            time_synchronizer.registerCallback(self.__synchronized_callback)
        else:
            labeled_points_callbacks = [self.__labeled_points_callback_1,
                                        self.__labeled_points_callback_2]
            self.__subscribers.append(rospy.Subscriber(rtabmap_grid_topic, OccupancyGrid, self.__rtabmap_callback))
            for i in range(self.cameras_number):
                self.__subscribers.append(rospy.Subscriber(labeled_points_topic + str(i + 1),
                                                           PointsList,
                                                           labeled_points_callbacks[i]))

        self.rtabmap_grid_map = None
        self.labeled_points = [None, None]
        self.index = 0

    def __synchronized_callback(self, *args):
        self.rtabmap_grid_map = args[0]

        for i in range(self.cameras_number):
            self.labeled_points[i] = args[i + 1]

        # self.index += 1
        # print(self.index)

    def __rtabmap_callback(self, rtabmap):
        self.rtabmap_grid_map = rtabmap

    def __labeled_points_callback_1(self, labeled_points):
        self.labeled_points[0] = labeled_points

        self.index += 1
        # print(self.index)

    def __labeled_points_callback_2(self, labeled_points):
        self.labeled_points[1] = labeled_points

    def get_labeled_point_cloud(self):
        # labeled_points_data = []
        labels_data = []
        points_data = []
        for labeled_points in self.labeled_points[:self.cameras_number]:
            if labeled_points is None:
                return None, None
            # Extracting variables
            points = labeled_points.point_cloud
            labels = labeled_points.labels

            # Obtaining point cloud
            pc = ros_numpy.numpify(points)
            points = np.zeros((pc.shape[0], 3))

            points[:, 0] = pc['x'].reshape(-1, )
            points[:, 1] = pc['y'].reshape(-1, )
            points[:, 2] = pc['z'].reshape(-1, )

            points_data.append(points.astype(np.float16))
            labels_data.append(np.array(labels).astype(np.int16).reshape(-1, 1))

        points = np.vstack(points_data)
        labels = np.vstack(labels_data)

        return points, labels

    def get_rtabmap_grid(self):
        if self.rtabmap_grid_map is None:
            return None, None

        height = self.rtabmap_grid_map.info.height
        width = self.rtabmap_grid_map.info.width
        origin = self.rtabmap_grid_map.info.origin.position
        rtabmap_grid = np.array(self.rtabmap_grid_map.data).reshape(height, width)

        return rtabmap_grid, origin
