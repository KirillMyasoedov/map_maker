import numpy as np
import rospy
from time import time
from PIL import Image
import os


class MapMakingNode(object):
    def __init__(self, input_map_adapter, occupancy_map_maker, output_map_adapter, period=0.1):
        # Receiving submodules
        self.__input_map_adapter = input_map_adapter
        self.__occupancy_map_maker = occupancy_map_maker
        self.__output_map_adapter = output_map_adapter

        # Initializing loop
        self.__timer = rospy.Timer(rospy.Duration(period), self.__timer_callback)

        # Maps after creating
        self.save_maps_image = True
        self.calculate_maps_accuracy = True

    def __timer_callback(self, _):
        start_time = time()
        # Receiving of data
        points, labels = self.__input_map_adapter.get_labeled_point_cloud()
        rtabmap_grid, rtabmap_origin = self.__input_map_adapter.get_rtabmap_grid()

        index = self.__input_map_adapter.index

        if labels is None or points is None or rtabmap_grid is None:
            return None
        # print("messages receiving rate", time() - start_time)
        # current_time = time()
        # Obtaining of grid map
        non_passable_grid, passable_grid, points, labels = self.__occupancy_map_maker.make_map(labels,
                                                                                               points,
                                                                                               rtabmap_origin,
                                                                                               save_history_files=False,
                                                                                               return_point_cloud=True,
                                                                                               save_history=False)
        non_passable_grid, passable_grid = self.__occupancy_map_maker.combine_with_rtabmap(non_passable_grid,
                                                                                           passable_grid,
                                                                                           rtabmap_grid,
                                                                                           rtabmap_origin,
                                                                                           save_combined_map=False)
        if self.save_maps_image:
            self.__occupancy_map_maker.save_maps(passable_grid, non_passable_grid)
        if self.calculate_maps_accuracy:
            self.__occupancy_map_maker.calculate_accuracy(passable_grid, non_passable_grid)

        # print("map making rate", time() - current_time)
        # current_time = time()
        height = self.__occupancy_map_maker.height
        width = self.__occupancy_map_maker.width
        resolution = self.__occupancy_map_maker.resolution
        origin_pose = self.__occupancy_map_maker.origin_pose

        # Publishing of data
        self.__output_map_adapter.publish_map(non_passable_grid,
                                              passable_grid,
                                              height,
                                              width,
                                              resolution,
                                              origin_pose,
                                              points,
                                              labels,
                                              visualize_point_cloud=True)
        # print("publishing rate", time() - current_time)
        print('Map making node rate:', time() - start_time)
