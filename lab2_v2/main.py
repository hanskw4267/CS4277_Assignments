""" CS4277/CS5477 Lab 2: Metric Rectification and Robust Homography Estimation.
See accompanying file (lab2.pdf) for instructions.

Name: Hans Kurnia
Email: e0310940@u.nus.edu
Student ID: A0184145E
"""

import numpy as np
import cv2
from helper import *
from math import floor, ceil, sqrt


def compute_homography(src, dst):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """

    h_matrix = np.eye(3, dtype=np.float64)

    """ YOUR CODE STARTS HERE """
    # Compute normalization matrix
    centroid_src = np.mean(src, axis=0)
    d_src = np.linalg.norm(src - centroid_src[None, :], axis=1)
    s_src = sqrt(2) / np.mean(d_src)
    T_norm_src = np.array([[s_src, 0.0, -s_src * centroid_src[0]],
                           [0.0, s_src, -s_src * centroid_src[1]],
                           [0.0, 0.0, 1.0]])

    centroid_dst = np.mean(dst, axis=0)
    d_dst = np.linalg.norm(dst - centroid_dst[None, :], axis=1)
    s_dst = sqrt(2) / np.mean(d_dst)
    T_norm_dst = np.array([[s_dst, 0.0, -s_dst * centroid_dst[0]],
                           [0.0, s_dst, -s_dst * centroid_dst[1]],
                           [0.0, 0.0, 1.0]])

    srcn = transform_homography(src, T_norm_src)
    dstn = transform_homography(dst, T_norm_dst)

    # Compute homography
    n_corr = srcn.shape[0]
    A = np.zeros((n_corr*2, 9), dtype=np.float64)
    for i in range(n_corr):
        A[2 * i, 0] = srcn[i, 0]
        A[2 * i, 1] = srcn[i, 1]
        A[2 * i, 2] = 1.0
        A[2 * i, 6] = -dstn[i, 0] * srcn[i, 0]
        A[2 * i, 7] = -dstn[i, 0] * srcn[i, 1]
        A[2 * i, 8] = -dstn[i, 0] * 1.0

        A[2 * i + 1, 3] = srcn[i, 0]
        A[2 * i + 1, 4] = srcn[i, 1]
        A[2 * i + 1, 5] = 1.0
        A[2 * i + 1, 6] = -dstn[i, 1] * srcn[i, 0]
        A[2 * i + 1, 7] = -dstn[i, 1] * srcn[i, 1]
        A[2 * i + 1, 8] = -dstn[i, 1] * 1.0

    u, s, vt = np.linalg.svd(A)
    h_matrix_n = np.reshape(vt[-1, :], (3, 3))

    # Unnormalize homography
    h_matrix = np.linalg.inv(T_norm_dst) @ h_matrix_n @ T_norm_src
    h_matrix /= h_matrix[2, 2]

    # src = src.astype(np.float32)
    # dst = dst.astype(np.float32)
    # h_matrix = cv2.findHomography(src, dst)[0].astype(np.float64)
    """ YOUR CODE ENDS HERE """

    return h_matrix


def transform_homography(src, h_matrix):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    Prohibited functions:
        cv2.perspectiveTransform()

    """
    transformed = None

    """ YOUR CODE STARTS HERE """
    transformed = np.ones_like(src)
    for i, pt in enumerate(src):
        token = np.transpose(
            np.matmul(h_matrix, np.transpose(np.append(pt, [1]))))
        transformed[i] = token[:-1]/token[-1]
    """ YOUR CODE ENDS HERE """

    return transformed


def warp_image(src, dst, h_matrix):
    """Applies perspective transformation to source image to warp it onto the
    destination (background) image

    Args:
        src (np.ndarray): Source image to be warped
        dst (np.ndarray): Background image to warp template onto
        h_matrix (np.ndarray): Warps coordinates from src to the dst, i.e.
                                 x_{dst} = h_matrix * x_{src},
                               where x_{src}, x_{dst} are the homogeneous
                               coordinates in I_{src} and I_{dst} respectively

    Returns:
        dst (np.ndarray): Source image warped onto destination image

    Prohibited functions:
        cv2.warpPerspective()
    You may use the following functions: np.meshgrid(), cv2.remap(), transform_homography()
    """
    dst = dst.copy()  # deep copy to avoid overwriting the original image

    """ YOUR CODE STARTS HERE """
    h, w = dst.shape[:-1]
    M = []
    for x in range(w):
        for y in range(h):
            M.append([x, y])
    M = np.array(M)
    transformed = transform_homography(M, np.linalg.inv(h_matrix))
    map_x = transformed[:, 0].reshape(h, w, order="F").astype(np.float32)
    map_y = transformed[:, 1].reshape(h, w, order="F").astype(np.float32)

    dst = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_TRANSPARENT, dst=dst)

    """ YOUR CODE ENDS HERE """
    # cv2.warpPerspective(src, h_matrix, dsize=dst.shape[1::-1],
    #                     dst=dst, borderMode=cv2.BORDER_TRANSPARENT)
    return dst


def compute_affine_rectification(src_img: np.ndarray, lines_vec: list):
    '''
       The first step of the stratification method for metric rectification. Compute
       the projective transformation matrix Hp with line at infinity. At least two
       parallel line pairs are required to obtain the vanishing line. Then warping
       the image with the predicted projective transformation Hp to recover the affine
       properties. X_dst=Hp*X_src

       Args:
           src_img: Original image X_src
           lines_vec: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0).
           Every two neighboring lines forms a constraint,
           e.g., for lines_vec=[l1,l2,l3,l4], it should be l1//l2 and l3//l4.
       Returns:
           Xa: Affinely rectified image by removing projective distortion
       You may use functions in helper.py, e.g. class Line, class Point, etc.
    '''
    dst = np.zeros_like(
        src_img)  # deep copy to avoid overwriting the original image
    Hp = np.zeros((3, 3))
    """ YOUR CODE STARTS HERE """
    pt1 = lines_vec[0].intersetion_point(lines_vec[1])
    pt2 = lines_vec[2].intersetion_point(lines_vec[3])
    a, b, c = Line(pt1, pt2).vec_para
    Hp = np.linalg.inv(np.array([[1, 0, 0], [0, 1, 0], [-a/c, -b/c, 1/c]]))

    h, w = src_img.shape[:-1]
    M = []
    for x in range(w):
        for y in range(h):
            M.append([x, y])
    M = np.array(M)
    transformed = transform_homography(M, Hp)
    new_w, new_h = np.max(transformed, axis=0)
    dst = np.zeros((ceil(new_h), ceil(new_w), 3))

    dst = warp_image(src_img, dst, Hp)
    """ YOUR CODE ENDS HERE """

    return dst


def compute_metric_rectification_step2(src_img: np.ndarray, line_vecs: list):
    '''
       The second step of the stratification method for metric rectification. Compute
       the affine transformation Ha with the degenerate conic from at least two
       orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=Ha*X_src

       Args:
           src_img: Affinely rectified image X_src
           line_vecs: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
           Every two neighboring lines forms a constraint,
           e.g., for lines_vec=[l1,l2,l3,l4], it should be l1 \perp l2 and l3 \perp l4.
       Returns:
           X_dst: Image after metric rectification
        You may use functions in helper.py, e.g. class Line, class Point, etc.
    '''
    dst = np.zeros_like(
        src_img)  # deep copy to avoid overwriting the original image
    Ha = np.zeros((3, 3))
    """ YOUR CODE STARTS HERE """
    constraints = []
    for i in range(0, len(line_vecs), 2):
        l1, l2, l3 = line_vecs[i].vec_para
        m1, m2, m3 = line_vecs[i+1].vec_para
        constraints.append([l1*m1, (l1*m2 + l2*m1), l2*m2])
    constraints = np.array(constraints)

    u, s, vh = np.linalg.svd(constraints)
    null_vec = vh[-1, :]
    S = np.array([[null_vec[0], null_vec[1]], [null_vec[1], null_vec[2]]])
    K = np.linalg.cholesky(S).transpose()
    K = K / np.linalg.det(K)
    Ha = np.linalg.inv(
        np.array([np.append(K[0, :], 0), np.append(K[1, :], 0), [0, 0, 1]]))

    h, w = src_img.shape[:-1]
    M = []
    for x in range(w):
        for y in range(h):
            M.append([x, y])
    M = np.array(M)
    transformed = transform_homography(M, Ha)
    new_w, new_h = np.max(transformed, axis=0)
    dst = np.zeros((ceil(new_h), ceil(new_w), 3))

    dst = warp_image(src_img, dst, Ha)
    """ YOUR CODE ENDS HERE """

    return dst


def compute_metric_rectification_one_step(src_img: np.ndarray, line_vecs: list):
    '''
       One-step metric rectification. Compute the transformation matrix H (i.e. H=HaHp) directly
       from five orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=H*X_src
       Args:
           src_img: Original image Xc
           lines_vec: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0).
           Every two neighboring lines forms a constraint,
           e.g., for lines_vec=[l1,l2,l3,l4,...,l10], it should be l1 \perp l2, l3 \perp l4,
           ..., l9 \perp l10.
       Returns:
           Xa: Image after metric rectification
        You may use functions in helper.py, e.g. class Line, class Point, etc.
    '''
    dst = np.zeros_like(
        src_img)  # deep copy to avoid overwriting the original image
    H = np.zeros((3, 3))

    """ YOUR CODE STARTS HERE """
    constraints = []
    for i in range(0, len(line_vecs), 2):
        l1, l2, l3 = line_vecs[i].vec_para
        m1, m2, m3 = line_vecs[i+1].vec_para
        constraints.append([l1*m1, (l1*m2 + l2*m1)/2, l2*m2,
                            (l1*m3 + l3*m1)/2, (l2*m3 + l3*m2)/2, l3*m3])
    constraints = np.array(constraints)

    u, s, vh = np.linalg.svd(constraints)
    a, b, c, d, e, f = vh[-1, :]
    C = np.array([[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, f]])

    c_u, c_s, c_vh = np.linalg.svd(C)
    c_s = np.append(c_s[:-1], [1])
    c_s = np.sqrt(c_s)
    D = np.array([[c_s[0], 0, 0], [0, c_s[1], 0], [0, 0, c_s[2]]])
    H = np.linalg.inv(np.matmul(c_u, D))

    h, w = src_img.shape[:-1]
    M = []
    for x in range(w):
        for y in range(h):
            M.append([x, y])
    M = np.array(M)
    transformed = transform_homography(M, H)
    x_min, y_min = np.min(transformed, axis=0)
    x_max, y_max = np.max(transformed, axis=0)
    sx = w/(x_max - x_min)
    sy = h/(y_max - y_min)
    sim_trans = np.array(
        [[sx, 0, abs(x_min)*sx], [0, sy, abs(y_min)*sy], [0, 0, 1]])
    H = np.matmul(sim_trans, H)

    dst = warp_image(src_img, dst, H)
    """ YOUR CODE ENDS HERE """

    return dst


def compute_homography_error(src, dst, homography):
    """Compute the squared bidirectional pixel reprojection error for
    provided correspondences

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        homography (np.ndarray): Homography matrix that transforms src to dst.

    Returns:
        err (np.ndarray): Array of size (N, ) containing the error d for each
        correspondence, computed as:
          d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
        where ||a|| denotes the l2 norm (euclidean distance) of vector a.
    """
    d = np.zeros(src.shape[0], np.float64)

    """ YOUR CODE STARTS HERE """
    est_dst = transform_homography(src, homography)
    est_src = transform_homography(dst, np.linalg.inv(homography))
    for i in range(len(src)):
        d[i] = np.linalg.norm(dst[i] - est_dst[i])**2 + \
            np.linalg.norm(src[i] - est_src[i])**2
    """ YOUR CODE ENDS HERE """

    return d


def compute_homography_ransac(src, dst, thresh=16.0, num_tries=200):
    """Calculates the perspective transform from at least 4 points of
    corresponding points in a robust manner using RANSAC. After RANSAC, all the
    inlier correspondences will be used to re-estimate the homography matrix.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        thresh (float): Maximum allowed squared bidirectional pixel reprojection
          error to treat a point pair as an inlier (default: 16.0). Pixel
          reprojection error is computed as:
            d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
          where ||a|| denotes the l2 norm (euclidean distance) of vector a.
        num_tries (int): Number of trials for RANSAC

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.
        mask (np.ndarraay): Output mask with dtype np.bool where 1 indicates
          inliers

    Prohibited functions:
        cv2.findHomography()
    """

    h_matrix = np.eye(3, dtype=np.float64)
    mask = np.ones(src.shape[0], dtype=np.bool)

    """ YOUR CODE STARTS HERE """
    max_inliners = 0

    for n in range(num_tries):
        chosen_pts = np.random.randint(0, len(src), size=4)
        H = compute_homography(src[chosen_pts, :], dst[chosen_pts, :])
        D = compute_homography_error(src, dst, H)
        inliners = np.sum(np.where(D < thresh, True, False))
        if inliners > max_inliners:
            max_inliners = inliners
            h_matrix = H
            mask = np.where(D < thresh, True, False)
    h_matrix = compute_homography(src[mask], dst[mask])

    """ YOUR CODE ENDS HERE """

    return h_matrix, mask
