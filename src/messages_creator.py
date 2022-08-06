from sensor_msgs.msg import PointField
from sensor_msgs import point_cloud2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from point_cloud_classifier.msg import PointsList
import numpy as np
from time import time


def create_point_cloud_message(points, labels=None, header=None):
    if labels is None:
        point_cloud = points.tolist()
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
    else:
        rgb = np.ones((labels.shape[0], 3), dtype=np.uint32) * 255
        non_passable_mask = np.where(labels // 1000 == 1)
        passable_mask = np.where(labels // 1000 == 2)

        rgb[non_passable_mask, 0] = (labels[non_passable_mask] - 1000) * 8 + 15
        rgb[passable_mask, 1] = (labels[passable_mask] - 2000) * 8 + 15
        rgb_uint = ((rgb[:, 0] << 16) + (rgb[:, 1] << 8) + (rgb[:, 2]))

        point_cloud = np.zeros((points.shape[0], 4), dtype=object)
        point_cloud[:, :3] = points
        point_cloud[:, 3] = rgb_uint
        point_cloud = point_cloud.tolist()

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.UINT32, 1)]

    if header is None:
        header = Header()

    header.frame_id = "map"

    point_cloud_message = point_cloud2.create_cloud(header, fields, point_cloud)

    return point_cloud_message


def create_occupancy_grid_message(header, grid_array, height, width, resolution, origin_pose):
    # Creating data list
    grid_data = grid_array.flatten().tolist()

    occupancy_grid = OccupancyGrid()
    occupancy_grid.header = header
    occupancy_grid.info.height = height
    occupancy_grid.info.width = width
    occupancy_grid.info.resolution = resolution
    occupancy_grid.info.origin.position.x = origin_pose[0]
    occupancy_grid.info.origin.position.y = origin_pose[1]
    occupancy_grid.info.origin.position.z = origin_pose[2]
    occupancy_grid.data = grid_data

    return occupancy_grid


def create_labeled_points_message(header, point_cloud, labels):
    points_message = PointsList()
    points_message.header = header
    points_message.point_cloud = point_cloud
    points_message.labels = labels.flatten().astype(np.int16).tolist()

    return points_message


def create_labels_message(header, labels):
    labels = labels.astype(np.int16).flatten().tolist()
    labels_message = PointsList()
    labels_message.header = header
    labels_message.labels = labels

    return labels_message
