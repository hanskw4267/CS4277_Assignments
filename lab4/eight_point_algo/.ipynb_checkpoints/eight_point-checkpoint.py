""" CS4277/CS5477 Lab 4-1: Relative pose estimation with 8-point algorithm
See accompanying file (eightpoint.pdf) for instructions.

Name: Hans Kurnia
Email: e0310940@u.nus.edu
Student ID: A0184145E
"""

import numpy as np
import scipy.io as sio
import h5py
import cv2
import matplotlib.pyplot as plt
from math import sqrt


"""Helper functions: You should not have to touch the following functions.
"""


def compute_right_epipole(F):

    U, S, V = np.linalg.svd(F.T)
    e = V[-1]
    return e / e[2]


def plot_epipolar_line(img1, img2, F, x1, x2, epipole=None, show_epipole=False):
    """
    Visualize epipolar lines in the imame

    Args:
        img1, img2: two images from different views
        F: fundamental matrix
        x1, x2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate
    Returns:

    """
    plt.figure()
    plt.imshow(img1)
    for i in range(x1.shape[1]):
        plt.plot(x1[0, i], x1[1, i], 'bo')
        m, n = img1.shape[:2]
        line1 = np.dot(F.T, x2[:, i])
        t = np.linspace(0, n, 100)
        lt1 = np.array([(line1[2] + line1[0] * tt) / (-line1[1]) for tt in t])
        ndx = (lt1 >= 0) & (lt1 < m)
        plt.plot(t[ndx], lt1[ndx], linewidth=2)
    plt.figure()
    plt.imshow(img2)

    for i in range(x2.shape[1]):
        plt.plot(x2[0, i], x2[1, i], 'ro')
        if show_epipole:
            if epipole is None:
                epipole = compute_right_epipole(F)
            plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')

        m, n = img2.shape[:2]
        line2 = np.dot(F, x1[:, i])

        t = np.linspace(0, n, 100)
        lt2 = np.array([(line2[2] + line2[0] * tt) / (-line2[1]) for tt in t])

        ndx = (lt2 >= 0) & (lt2 < m)
        plt.plot(t[ndx], lt2[ndx], linewidth=2)
    plt.show()


def compute_essential(data1, data2, K):
    """
    Compute the essential matrix from point correspondences and intrinsic matrix

    Args:
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate
        K: intrinsic matrix of the camera
    Returns:
        E: Essential matrix
    """

    """YOUR CODE STARTS HERE"""
    data1 = data1[:2, :].T
    data2 = data2[:2, :].T

    # Normalize points
    def normalize_pts(data, T_matrix):
        norm_data = np.ones_like(data)
        for i, pt in enumerate(data):
            token = np.transpose(
                np.matmul(T_matrix, np.transpose(np.append(pt, [1]))))
            norm_data[i] = token[:-1]/token[-1]
        return norm_data

    norm_data1 = normalize_pts(data1, np.linalg.inv(K))
    norm_data2 = normalize_pts(data2, np.linalg.inv(K))

    # Find initial F matrix
    constraints = []
    for i in range(15):
        x, y = norm_data1[i]
        x_prime, y_prime = norm_data2[i]
        constraints.append([x_prime*x, x_prime*y, x_prime,
                            y_prime*x, y_prime*y, y_prime, x, y, 1])
    constraints = np.array(constraints)

    _, _, vh = np.linalg.svd(constraints)
    e11, e12, e13, e21, e22, e23, e31, e32, e33 = vh[-1, :]
    E = [[e11, e12, e13],
         [e21, e22, e23],
         [e31, e32, e33]]

    # enforce singularity constraint
    e_u, e_s, e_vh = np.linalg.svd(E)
    e_s[:2] = (e_s[0] + e_s[1]) / 2
    e_s[-1] = 0.0
    e_s = np.diag(e_s)
    E = np.matmul(np.matmul(e_u, e_s), e_vh)

    """YOUR CODE ENDS HERE"""

    return E


def decompose_e(E, K, data1, data2):
    """
    Compute the essential matrix from point correspondences and intrinsic matrix

    Args:
        E: Essential matrix
        K: intrinsic matrix of the camera
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate

    Returns:
        trans: 3x4 array representing the transformation matrix
    """
    """YOUR CODE STARTS HERE"""
    data1 = data1[:2, :].T
    data2 = data2[:2, :].T

    W = [[0, -1, 0],
         [1, 0, 0],
         [0, 0, 1]]
    np.array(W)

    e_u, e_s, e_vh = np.linalg.svd(E)

    t = e_u[:, 2].reshape((3, 1))
    r1 = np.matmul(np.matmul(e_u, W), e_vh)
    r2 = np.matmul(np.matmul(e_u, np.transpose(W)), e_vh)

    if (np.linalg.det(r1) < 0):
        r1 = -r1
    if (np.linalg.det(r2) < 0):
        r2 = -r2

    possible_p_prime = []
    possible_p_prime.append(np.concatenate([r1, t], axis=1))
    possible_p_prime.append(np.concatenate([r1, -t], axis=1))
    possible_p_prime.append(np.concatenate([r2, t], axis=1))
    possible_p_prime.append(np.concatenate([r2, -t], axis=1))

    I = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    P = np.hstack([I, [[0], [0], [0]]])

    # try each possible P prime using triangulation
    for i in range(4):
        A = [data1[0, 0] * P[2] - P[0],
             data1[0, 1] * P[2] - P[1],
             data2[0, 0] * possible_p_prime[i][2] - possible_p_prime[i][0],
             data2[0, 1] * possible_p_prime[i][2] - possible_p_prime[i][1]]
        A = np.array(A)

        _, _, vh = np.linalg.svd(A)
        X = vh[-1, :] / vh[-1, -1]

        # check if point X is infront of camera
        X_prime1 = np.matmul(possible_p_prime[i], X)
        X_prime2 = np.matmul(P, X)
        if(X_prime1[-1] > 0 and X_prime2[-1] > 0):
            trans = possible_p_prime[i]
            break

    """YOUR CODE ENDS HERE"""
    return trans


def compute_fundamental(data1, data2):
    """
    Compute the fundamental matrix from point correspondences

    Args:
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate

    Returns:
        F: fundamental matrix
    """

    """YOUR CODE STARTS HERE"""
    data1 = data1[:2, :].T
    data2 = data2[:2, :].T

    # Find normalizing matrices
    def find_T_matrix(data):
        centroid_data = np.mean(data, axis=0)
        d_data = np.linalg.norm(data1 - centroid_data[None, :], axis=1)
        s_data = sqrt(2) / np.mean(d_data)
        T = np.array([[s_data, 0.0, -s_data * centroid_data[0]],
                      [0.0, s_data, -s_data * centroid_data[1]],
                      [0.0, 0.0, 1.0]])
        return T

    T_norm_data1 = find_T_matrix(data1)
    T_norm_data2 = find_T_matrix(data2)

    # Normalize points
    def normalize_pts(data, T_matrix):
        norm_data = np.ones_like(data)
        for i, pt in enumerate(data):
            token = np.transpose(
                np.matmul(T_matrix, np.transpose(np.append(pt, [1]))))
            norm_data[i] = token[:-1]/token[-1]
        return norm_data

    norm_data1 = normalize_pts(data1, T_norm_data1)
    norm_data2 = normalize_pts(data2, T_norm_data2)

    # Find initial F matrix
    constraints = []
    for i in range(15):
        x, y = norm_data1[i]
        x_prime, y_prime = norm_data2[i]
        constraints.append([x_prime*x, x_prime*y, x_prime,
                            y_prime*x, y_prime*y, y_prime, x, y, 1])
    constraints = np.array(constraints)

    _, _, vh = np.linalg.svd(constraints)
    f11, f12, f13, f21, f22, f23, f31, f32, f33 = vh[-1, :]
    F = [[f11, f12, f13],
         [f21, f22, f23],
         [f31, f32, f33]]

    # enforce singularity constraint
    f_u, f_s, f_vh = np.linalg.svd(F)
    f_s[-1] = 0.0
    f_s = np.diag(f_s)
    F_prime = np.matmul(np.matmul(f_u, f_s), f_vh)
    F = np.matmul(np.matmul(T_norm_data2.T, F_prime), T_norm_data1)
    F = F / F[-1, -1]

    """YOUR CODE ENDS HERE"""

    return F
