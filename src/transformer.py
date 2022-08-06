from tf.transformations import *
import numpy as np


def transform_point(tf_data, point):
    T = translation_matrix((tf_data[0][0], tf_data[0][1], tf_data[0][2]))
    R = quaternion_matrix(tf_data[1])
    M = concatenate_matrices(T, R)

    point = np.vstack((point, 1))
    point = np.dot(M, point)

    return point[:3]


def transform_pcl(tf_data, points):
    T = translation_matrix((tf_data[0][0], tf_data[0][1], tf_data[0][2]))
    R = quaternion_matrix(tf_data[1])
    M = concatenate_matrices(T, R)

    ones = np.ones((points.shape[0], 1))
    points = np.concatenate((points, ones), axis=1)
    points = np.dot(M, points.T)

    return points[:3, :].T


def transform_plane(tf_data, normal, point):
    T = translation_matrix((tf_data[0][0], tf_data[0][1], tf_data[0][2]))
    R = quaternion_matrix(tf_data[1])
    M = concatenate_matrices(T, R)
    n_t = np.dot(R, np.vstack((normal, 1)))
    p_t = np.dot(M, np.vstack((point, 1)))
    A = n_t[0]
    B = n_t[1]
    C = n_t[2]
    D = - n_t[0] * p_t[0] - n_t[1] * p_t[1] - n_t[2] * p_t[2]

    return A, B, C, D
