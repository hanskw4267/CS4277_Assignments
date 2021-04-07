""" CS4277/CS5477 Lab 4-2: Absolute Pose Estimation
See accompanying file (pnp.pdf) for instructions.

Name: Hans Kurnia
Email: e0310940@u.nus.edu
Student ID: A0184145E
"""

import numpy as np
import cv2
from sympy.polys import subresultants_qq_zz
import sympy as sym
from math import sqrt
from cmath import acos
from itertools import combinations


"""Helper functions: You should not have to touch the following functions.
"""


def extract_coeff(x1, x2, x3, cos_theta12, cos_theta23, cos_theta13, d12, d23, d13):
    """
    Extract coefficients of a polynomial

    Args:
        x1, x2, x3: symbols representing the unknown camera-object distance
        cos_theta12, cos_theta23, cos_theta13: cos values of the inter-point angles
        d12, d23, d13: square of inter-point distances

    Returns:
        a: the coefficients of the polynomial of x1
    """
    f12 = x1 ** 2 + x2 ** 2 - 2 * x1 * x2 * cos_theta12 - d12
    f23 = x2 ** 2 + x3 ** 2 - 2 * x2 * x3 * cos_theta23 - d23
    f13 = x1 ** 2 + x3 ** 2 - 2 * x1 * x3 * cos_theta13 - d13
    matrix = subresultants_qq_zz.sylvester(f23, f13, x3)
    f12_ = matrix.det()
    f1 = subresultants_qq_zz.sylvester(f12, f12_, x2).det()
    a1 = f1.func(*[term for term in f1.args if not term.free_symbols])
    a2 = f1.coeff(x1 ** 2)
    a3 = f1.coeff(x1 ** 4)
    a4 = f1.coeff(x1 ** 6)
    a5 = f1.coeff(x1 ** 8)
    a = np.array([a1, a2, a3, a4, a5])
    return a


def icp(points_s, points_t):
    """
    Estimate the rotation and translation using icp algorithm

    Args:
        points_s : 10 x 3 array containing 3d points in the world coordinate
        points_t : 10 x 3 array containing 3d points in the camera coordinate

    Returns:
        r: rotation matrix of the camera
        t: translation of the camera
    """
    us = np.mean(points_s, axis=0, keepdims=True)
    ut = np.mean(points_t, axis=0, keepdims=True)
    points_s_center = points_s - us
    points_t_center = points_t - ut
    w = np.dot(points_s_center.T, points_t_center)
    u, s, vt = np.linalg.svd(w)
    r = vt.T.dot(u.T)
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T.dot(u.T)
    t = ut.T - np.dot(r, us.T)
    return r, t


def reconstruct_3d(X, K, points2d):
    """
    Reconstruct the 3d points from camera-point distance

    Args:
        X: a list containing camera-object distances for all points
        K: intrinsics of camera
        points2d: 10x1x3 array containing 2d coordinates of points in the homogeneous coordinate

    Returns:
        points3d_c: 3d coordinates of all points in the camera coordinate
    """
    points3d_c = []
    for i in range(len(X)):
        points3d_c.append(X[i] * np.dot(np.linalg.inv(K), points2d[i].T))
    points3d_c = np.hstack(points3d_c)
    return points3d_c


def visualize(r, t, points3d, points2d, K):
    """
    Visualize reprojections of all 3d points in the image and compare with ground truth

    Args:
        r: rotation matrix of the camera
        t: tranlation of the camera
        points3d:  10x3 array containing 3d coordinates of points in the world coordinate
        points3d:  10x2 array containing ground truth 2d coordinates of points in the image space
    """
    scale = 0.2
    img = cv2.imread('data/img_id4_ud.JPG')
    dim = (int(img.shape[1]*scale), int(img.shape[0]*scale))
    img = cv2.resize(img, dim)
    trans = np.hstack([r, t])
    points3d_homo = np.hstack([points3d, np.ones((points3d.shape[0], 1))])
    points2d_re = np.dot(K, np.dot(trans, points3d_homo.T))
    points2d_re = np.transpose(points2d_re[:2, :]/points2d_re[2:3, :])
    for j in range(points2d.shape[0]):
        cv2.circle(img, (int(points2d[j, 0]*scale),
                         int(points2d[j, 1]*scale)), 3,  (0, 0, 255))
        cv2.circle(img, (int(points2d_re[j, 0]*scale),
                         int(points2d_re[j, 1]*scale)), 4,  (255, 0, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)


def pnp_algo(K, points2d, points3d):
    """
    Estimate the rotation and translation of camera by using pnp algorithm

    Args:
        K: intrinsics of camera
        points2d: 10x1x2 array containing 2d coordinates of points in the image space
        points2d: 10x1x3 array containing 3d coordinates of points in the world coordinate
    Returns:
        r: 3x3 array representing rotation matrix of the camera
        t: 3x1 array representing translation of the camera
    """
    """YOUR CODE STARTS HERE"""
    x1, x2, x3 = sym.symbols('x1, x2, x3')

    # Homogenize 2d points
    points2d_homo = np.ones((10, 1, 3))
    for i, pt in enumerate(points2d):
        points2d_homo[i][0][:2] = pt

    def calc_cos_theta(p1, p2):
        k_matrix = np.matmul(np.linalg.inv(K.T), np.linalg.inv(K))
        num = np.matmul(np.matmul(p1.T, k_matrix), p2)
        den1 = sqrt(np.matmul(np.matmul(p1.T, k_matrix), p1))
        den2 = sqrt(np.matmul(np.matmul(p2.T, k_matrix), p2))
        return num/den1/den2

    def calc_dist(p1, p2):
        return (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2

    X = []
    for i in range(10):
        constraints = []
        pt_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        pt_indices.remove(i)
        pt_1_2d = points2d_homo[i][0]
        pt_1_3d = points3d[i][0]
        points23 = combinations(pt_indices, 2)

        for j, k in list(points23):
            pt_2_3d = points3d[j][0]
            pt_2_2d = points2d_homo[j][0]
            pt_3_3d = points3d[k][0]
            pt_3_2d = points2d_homo[k][0]
            cos_theta12 = calc_cos_theta(pt_1_2d, pt_2_2d)
            cos_theta23 = calc_cos_theta(pt_2_2d, pt_3_2d)
            cos_theta13 = calc_cos_theta(pt_1_2d, pt_3_2d)
            d12 = calc_dist(pt_1_3d, pt_2_3d)
            d23 = calc_dist(pt_2_3d, pt_3_3d)
            d13 = calc_dist(pt_1_3d, pt_3_3d)
            constraints.append(extract_coeff(
                x1, x2, x3, cos_theta12, cos_theta23, cos_theta13, d12, d23, d13))
        A = np.array(constraints, dtype=np.float)
        u, s, vh = np.linalg.svd(A)
        t0, t1, t2, t3, t4 = vh[-1, :]
        X.append(sqrt(np.mean([t1/t0, t2/t1, t3/t2, t4/t3])))

    points3d_c = reconstruct_3d(X, K, points2d_homo)
    r, t = icp(np.squeeze(points3d), points3d_c.T)

    """YOUR CODE ENDS HERE"""
    return r, t
