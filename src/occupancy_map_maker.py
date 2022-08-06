import numpy as np
from .transformer import transform_pcl
import os
from PIL import Image
from .accuracies_calculator import AccuracyCalculator
import torch
import rospy
from time import time


def points_to_probs_grid(xs, ys, ws, grid):
    ps = np.ones_like(xs)

    xt = torch.from_numpy(xs).long()
    yt = torch.from_numpy(ys).long()
    pt = torch.from_numpy(ps).long()
    gridt = torch.from_numpy(grid).long()
    gridt.index_put_((xt, yt), pt, accumulate=True)

    grid = gridt.numpy().astype(np.float64)
    grid[xt, yt] *= ws

    return grid


class OccupancyMapMaker(object):
    def __init__(self, probability=0.01):
        # Map parameters
        environment_type = rospy.search_param("environment")
        self.environment = rospy.get_param(environment_type)

        self.height = 290  # 350
        self.width = 295  # 350
        self.resolution = 0.05
        self.origin_pose = [-7.125, -7.6, 0]  # setup5 [-7.125, -7.6, 0] setup5_3d [-7.17, -7.6, 0]

        # self.height = 470
        # self.width = 460
        # self.resolution = 0.05
        # self.origin_pose = [-11.7, -12, 0]  # setup3

        # Defining passable and non-passable grids
        map_type_parameter = rospy.search_param("map_type")
        self.map_type = rospy.get_param(map_type_parameter)

        self.non_passable_grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.passable_grid = np.zeros((self.height, self.width), dtype=np.int8)

        if self.map_type == "probability_map":
            self.probability = probability
            probability_type_parameter = rospy.search_param("measurement_type")
            self.probability_type = rospy.get_param(probability_type_parameter)

            self.passable_loggrid = np.zeros((self.height, self.width))
            self.non_passable_loggrid = np.zeros((self.height, self.width))

        # Saver variables
        self.common_labeled_points = np.array([], dtype=np.float64).reshape((0, 4))
        self.common_points = []
        self.common_labels = []
        self.product_saver_path = "/home/kirill/catkin_ws/src/map_maker/product"

        self.i = 0
        self.filter_points = True

        # Calculation accuracies
        # Loading truth mask
        self.mask_path = '/home/kirill/catkin_ws/src/map_maker/masks'
        self.accuracy_calculator = AccuracyCalculator()

        if self.environment == 'flat':
            passable_image = Image.open(os.path.join(self.mask_path, 'passable_mask_flat.png'))
            non_passable_image = Image.open(os.path.join(self.mask_path, 'unpassable_mask_flat.png'))

            self.passable_mask = np.array(passable_image)
            self.non_passable_mask = np.array(non_passable_image)
        elif self.environment == 'unstructured':
            passable_image = Image.open(os.path.join(self.mask_path, 'passable_mask_unstructured.png'))
            non_passable_image = Image.open(os.path.join(self.mask_path, 'unpassable_mask_unstructured.png'))

            self.passable_mask = np.array(passable_image)
            self.non_passable_mask = np.array(non_passable_image)

    def project_point_cloud(self, points, labels):
        # Creating occupancy mask
        grid_x = np.zeros(points.shape[0], dtype=np.int64)
        grid_y = np.zeros(points.shape[0], dtype=np.int64)

        grid_x[:] = (points[:, 0] - self.origin_pose[0]) // self.resolution
        grid_y[:] = (points[:, 1] - self.origin_pose[1]) // self.resolution

        # Cutting the points according to grid shape
        mask = ((grid_x > 0) * (grid_x < self.width)) * ((grid_y > 0) * (grid_y < self.height))
        grid_x_cutted = grid_x[mask]
        grid_y_cutted = grid_y[mask]
        labels_cutted = labels[mask]

        return grid_x_cutted, grid_y_cutted, labels_cutted, mask

    def make_map(self,
                 labels,
                 points,
                 rtabmap_origin=None,
                 tf_data=None,
                 return_point_cloud=False,
                 save_history_files=False,
                 save_history=False):
        if tf_data is not None:
            points = transform_pcl(tf_data, points)

        # Filtering out non-detected points
        # above_zero_mask = (points[:, 2] >= 0.025)
        # non_stone_mask = (labels[:, 0] == 0)
        # points = np.delete(points, np.where(above_zero_mask * non_stone_mask), axis=0)
        # labels = np.delete(labels, np.where(above_zero_mask * non_stone_mask), axis=0)

        grid_x_cutted, grid_y_cutted, labels_cutted, _ = self.project_point_cloud(points, labels)
        labels_cutted = labels_cutted // 1000

        # Creating passable and non-passable maps
        if self.map_type == "binary_map":
            self.non_passable_grid[grid_y_cutted[labels_cutted[:, 0] == 0], grid_x_cutted[labels_cutted[:, 0] == 0]] = 0
            self.non_passable_grid[grid_y_cutted[labels_cutted[:, 0] == 1], grid_x_cutted[labels_cutted[:, 0] == 1]] = 100
            self.passable_grid[grid_y_cutted[labels_cutted[:, 0] == 0], grid_x_cutted[labels_cutted[:, 0] == 0]] = 0
            self.passable_grid[grid_y_cutted[labels_cutted[:, 0] == 2], grid_x_cutted[labels_cutted[:, 0] == 2]] = 100

        if self.map_type == "probability_map":
            # Creating probability mask
            non_passable_probabilities = np.zeros((self.height, self.width))
            passable_probabilities = np.zeros((self.height, self.width))

            if self.probability_type == "per_cell_measurement":
                non_passable_probabilities[grid_y_cutted[labels_cutted[:, 0] == 0],
                                           grid_x_cutted[labels_cutted[:, 0] == 0]] = -1  # -1
                non_passable_probabilities[grid_y_cutted[labels_cutted[:, 0] == 1],
                                           grid_x_cutted[labels_cutted[:, 0] == 1]] = 0.5  # 0.5
                passable_probabilities[grid_y_cutted[labels_cutted[:, 0] == 0],
                                       grid_x_cutted[labels_cutted[:, 0] == 0]] = -1  # -1
                passable_probabilities[grid_y_cutted[labels_cutted[:, 0] == 2],
                                       grid_x_cutted[labels_cutted[:, 0] == 2]] = 10  # 10

                passable_probabilities[grid_y_cutted[labels_cutted[:, 0] == 1],
                                       grid_x_cutted[labels_cutted[:, 0] == 1]] = -1  # -1
            elif self.probability_type == "per_point_measurement":
                non_passable_probabilities = points_to_probs_grid(grid_y_cutted[labels_cutted[:, 0] == 0],
                                                                  grid_x_cutted[labels_cutted[:, 0] == 0],
                                                                  -0.05,
                                                                  non_passable_probabilities)
                non_passable_probabilities = points_to_probs_grid(grid_y_cutted[labels_cutted[:, 0] == 1],
                                                                  grid_x_cutted[labels_cutted[:, 0] == 1],
                                                                  0.009,
                                                                  non_passable_probabilities)
                passable_probabilities = points_to_probs_grid(grid_y_cutted[labels_cutted[:, 0] == 0],
                                                              grid_x_cutted[labels_cutted[:, 0] == 0],
                                                              -0.05,
                                                              passable_probabilities)
                passable_probabilities = points_to_probs_grid(grid_y_cutted[labels_cutted[:, 0] == 2],
                                                              grid_x_cutted[labels_cutted[:, 0] == 2],
                                                              0.15,
                                                              passable_probabilities)
                passable_probabilities = points_to_probs_grid(grid_y_cutted[labels_cutted[:, 0] == 1],
                                                              grid_x_cutted[labels_cutted[:, 0] == 1],
                                                              -0.02,
                                                              passable_probabilities)

            # Creating probability subgrid
            non_passable_subgrid = 0.5 * np.ones((self.height, self.width))
            passable_subgrid = 0.5 * np.ones((self.height, self.width))
            non_passable_subgrid += self.probability * non_passable_probabilities
            passable_subgrid += self.probability * passable_probabilities

            # Removing out of range probabilities
            non_passable_subgrid[non_passable_subgrid < 0] = 0
            passable_subgrid[passable_subgrid < 0] = 0
            non_passable_subgrid[non_passable_subgrid > 1] = 1
            passable_subgrid[passable_subgrid > 1] = 1

            # Calculation of logarithmic odd
            self.non_passable_loggrid += np.log(non_passable_subgrid / (1 - non_passable_subgrid))
            self.passable_loggrid += np.log(passable_subgrid / (1 - passable_subgrid))

            # Calculation of probability grid
            self.non_passable_grid = (1 / (1 + np.exp(-self.non_passable_loggrid)) * 100).astype(np.int64)
            self.passable_grid = (1 / (1 + np.exp(-self.passable_loggrid)) * 100).astype(np.int64)
            # self.non_passable_grid = ((1 - 1 / np.exp(self.non_passable_loggrid)) * 100).astype(np.int8)
            # self.passable_grid = ((1 - 1 / np.exp(self.passable_loggrid)) * 100).astype(np.int8)
            # print(np.unique(self.passable_subgrid))
            # print(np.unique(self.passable_grid))

            # Transforming probability map to binary map
            self.non_passable_grid[self.non_passable_grid <= 50] = 0
            self.non_passable_grid[self.non_passable_grid > 50] = 100
            self.passable_grid[self.passable_grid <= 50] = 0
            self.passable_grid[self.passable_grid > 50] = 100
            # print(np.unique(self.passable_grid))

        # Saving history
        if save_history_files:
            non_passable_mask = labels[:, 0] // 1000 == 1
            passable_mask = labels[:, 0] // 1000 == 2
            stone_labels = labels[passable_mask + non_passable_mask, 0].reshape(-1, 1)
            stone_points = points[passable_mask + non_passable_mask, :]

            labeled_points = np.hstack((stone_points.astype(np.float16),
                                        stone_labels.astype(np.int16)))
            self.common_labeled_points = np.vstack((self.common_labeled_points, labeled_points)).astype(np.float16)
            print(self.common_labeled_points.shape)

            if self.common_labeled_points.shape[0] >= 32084119:
                self.i += 1
                if self.i == 1:
                    if self.filter_points:
                        saver_points = self.common_labeled_points[:, :3]
                        saver_labels = self.common_labeled_points[:, 3].reshape(-1, 1)
                        grid_x, grid_y, saver_labels, mask = self.project_point_cloud(saver_points, saver_labels)

                        saver_points = saver_points[mask]

                        passable_cells = self.passable_grid[grid_y, grid_x]
                        non_passable_cells = self.non_passable_grid[grid_y, grid_x]

                        passable_mask = np.where(passable_cells == 100)
                        non_passable_mask = np.where(non_passable_cells == 100)

                        saver_points = np.vstack((saver_points[non_passable_mask],
                                                  saver_points[passable_mask])).astype(np.float16)
                        saver_labels = np.vstack((saver_labels[non_passable_mask],
                                                  saver_labels[passable_mask])).astype(np.int16)

                        common_labeled_points = np.hstack((saver_points, saver_labels)).astype(np.float16)
                    else:
                        common_labeled_points = self.common_labeled_points
                    if os.path.exists(os.path.join(self.product_saver_path, "labeled_points.npz")):
                        os.remove(os.path.join(self.product_saver_path, "labeled_points.npz"))
                    np.savez(os.path.join(self.product_saver_path, "labeled_points"), common_labeled_points)

        if save_history:
            self.common_points.append(points)
            self.common_labels.append(labels)
            points = np.vstack(self.common_points)
            labels = np.vstack(self.common_labels)
            
        if return_point_cloud:
            return self.non_passable_grid, self.passable_grid, points, labels
        else:
            return self.non_passable_grid, self.passable_grid

    def combine_with_rtabmap(self,
                             non_passable_grid,
                             passable_grid,
                             rtabmap_grid,
                             rtabmap_origin,
                             save_combined_map=False):
        # Removing passable pixels from rtabmap grid
        passable_mask = np.where(passable_grid > 50)
        x = passable_mask[1]
        y = passable_mask[0]

        x_rtab = x + int((self.origin_pose[0] - rtabmap_origin.x) // self.resolution)
        y_rtab = y + int((self.origin_pose[1] - rtabmap_origin.y) // self.resolution)

        # Cutting classifier_grid indices beyond rtabmap range
        filter_size = 7
        half_filter_size = filter_size // 2
        x_to_filter = x_rtab[((x_rtab < rtabmap_grid.shape[1] - half_filter_size) * (x_rtab > half_filter_size)) * \
                             ((y_rtab < rtabmap_grid.shape[0] - half_filter_size) * (y_rtab > half_filter_size))]
        y_to_filter = y_rtab[((x_rtab < rtabmap_grid.shape[1] - half_filter_size) * (x_rtab > half_filter_size)) * \
                             ((y_rtab < rtabmap_grid.shape[0] - half_filter_size) * (y_rtab > half_filter_size))]
        x_nonpass = x[((x < passable_grid.shape[1] - half_filter_size) * (x > half_filter_size)) * \
                      ((y < passable_grid.shape[0] - half_filter_size) * (y > half_filter_size))]
        y_nonpass = y[((x < passable_grid.shape[1] - half_filter_size) * (x > half_filter_size)) * \
                      ((y < passable_grid.shape[0] - half_filter_size) * (y > half_filter_size))]

        if x_to_filter.size and y_to_filter.size > 0:
            for i in range(filter_size):
                for j in range(filter_size):
                    rtabmap_grid[y_to_filter - half_filter_size + i, x_to_filter - half_filter_size + j] = 0
                    non_passable_grid[y_nonpass - half_filter_size + i, x_nonpass - half_filter_size + j] = 0

        # Adding non passables pixels to classifier grid
        non_passable_mask = np.where(rtabmap_grid > 50)
        x = non_passable_mask[1]
        y = non_passable_mask[0]

        x += int((rtabmap_origin.x - self.origin_pose[0]) // self.resolution)
        y += int((rtabmap_origin.y - self.origin_pose[1]) // self.resolution)

        # Cutting the points according to the grid shape
        mask = ((x > 0) * (x < non_passable_grid.shape[1])) * ((y > 0) * (y < non_passable_grid.shape[0]))

        non_passable_grid[y[mask], x[mask]] = 100
        if save_combined_map:
            np.savez(os.path.join(self.product_saver_path, "combined_non_passable_map"), non_passable_grid)
        return non_passable_grid, passable_grid

    def save_maps(self, passable_grid, non_passable_grid):
        passable_array = np.zeros((passable_grid.shape[0], passable_grid.shape[1], 3), dtype=np.uint8)
        non_passable_array = np.zeros((non_passable_grid.shape[0], non_passable_grid.shape[1], 3), dtype=np.uint8)

        passable_array[passable_grid == 100] = [255, 255, 255]
        non_passable_array[non_passable_grid == 100] = [255, 255, 255]

        passable_image = Image.fromarray(passable_array)
        non_passable_image = Image.fromarray(non_passable_array)

        passable_image.save(os.path.join(self.mask_path, 'passable_image.png'))
        non_passable_image.save(os.path.join(self.mask_path, 'non_passable_image.png'))

    def calculate_accuracy(self, passable_map, non_passable_map):
        # Passable map
        precision, recall, f1_score = \
            self.accuracy_calculator.calculate_image_accuracy(self.passable_mask, passable_map)
        print("Passable map precision: {}, recall: {}, f1_score: {}".format(precision, recall, f1_score))
        self.accuracy_calculator.save_comparison('passable_comparison', self.mask_path)

        # Non passable map
        precision, recall, f1_score = \
            self.accuracy_calculator.calculate_image_accuracy(self.non_passable_mask, non_passable_map)
        print("Unpassable map precision: {}, recall: {}, f1_score: {}".format(precision, recall, f1_score))
        self.accuracy_calculator.save_comparison('non_passable_comparison', self.mask_path)
