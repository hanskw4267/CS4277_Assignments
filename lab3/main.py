""" CS4277/CS5477 Lab 3: AFFINE 3D MEASUREMENT FROM VANISHING LINE AND POINT.
See accompanying file (lab3.pdf) for instructions.

Name: Hans Kurnia
Email: e0310940@u.nus.edu
Student ID: A0184145E
"""
import os
import copy
import argparse
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
RESULT_DIR = os.path.join(PROJECT_DIR, 'results')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Canny edge detector constants
MIN_CANNY_THRESHOLD = 100
MAX_CANNY_THRESHOLD = 200

# Hough Transform constants
MIN_LINE_LENGTH = 30
MAX_LINE_GAP = 5

DISTANCE_THRESHOLD = 5.


def detect_lines(image: np.ndarray) -> np.ndarray:
    """
    Detects lines in the image using Canny edge detector and Hough Line Transform

    Args:
        image: H x W x C image

    Returns:
        N x 4 line points (x1, y1, x2, y2)
    """
    line_pts = np.empty(shape=[0, 3], dtype=float)
    """ 
    YOUR CODE HERE
        1. Call cv2.Canny with threshold1=MIN_CANNY_THRESHOLD, threshold2=MAX_CANNY_THRESHOLD and apertureSize=3 
            to detect edges 
        2. Pass the edge image from Canny edge detection to cv2.HoughLinesP with rho=1, theta=np.pi/180, threshold=100,
            minLineLength=MIN_LINE_LENGTH and maxLineGap=MAX_LINE_GAP.
    """
    edge_image = cv2.Canny(image=image, threshold1=MIN_CANNY_THRESHOLD,
                           threshold2=MAX_CANNY_THRESHOLD, apertureSize=3)
    line_pts = cv2.HoughLinesP(image=edge_image, rho=1, theta=(
        np.pi/180), threshold=100, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    """ END YOUR CODE HERE """
    line_pts = line_pts.reshape(-1, 4)
    return line_pts


def _save_line_image(image: np.ndarray, line_pts: np.ndarray, save_prefix: os.path):
    """
    Saves the line image for visualization

    Args:
        image: H x W x C image
        line_pts: N x 4 line points (x1, y1, x2, y2)
        save_prefix: the prefix for the save file.
    """
    save_image = copy.deepcopy(image)
    for (x1, y1, x2, y2) in line_pts:
        save_image = cv2.line(save_image, (x1, y1),
                              (x2, y2), (255, 0, 0), thickness=2)
    save_image_file = save_prefix + '.jpg'
    cv2.imwrite(save_image_file, save_image)


def _get_lines_from_line_pts(line_pts: np.ndarray) -> np.ndarray:
    """
    Converts line points to line representation [a, b, 1] where ax + by + 1 = 0 and (x, y) corresponds to points [x,y,1]
    that lie on the line.

    Args:
        line_pts: N x 4 line points (x1, y1, x2, y2)

    Returns:
        L x 3 lines (a, b, 1)
    """
    assert line_pts.shape[1] == 4, 'shape should be N x 4'
    num_lines = line_pts.shape[0]
    pts = np.ones([num_lines, 3], dtype=float)
    other_pts = np.ones([num_lines, 3], dtype=float)
    pts[:, :2] = line_pts[:, :2]
    other_pts[:, :2] = line_pts[:, 2:]

    lines = np.cross(a=pts, b=other_pts)
    lines = lines / lines[:, -1].reshape(-1, 1)
    return lines


def get_pairwise_intersections(lines: np.ndarray) -> np.ndarray:
    """
    Computes intersections between each pair of lines. Remove non-existent intersections i.e. intersections at
    infinity where z=0 for points [x,y,z]

    Args:
        lines: L x 3 lines (a, b, 1)

    Returns:
        I x 3 pairwise intersections [u, v, 1] where I <= N^2
    """
    num_lines = lines.shape[0]
    intersections = np.empty(shape=[num_lines**2, 3], dtype=lines.dtype)
    """ YOUR CODE HERE """
    intersections[:, :] = 0
    count = 0
    for i in range(num_lines):
        for k in range(num_lines):
            intersection = np.cross(lines[i], lines[k])
            if (intersection[2] != 0):
                intersections[count] = intersection/intersection[2]
                count += 1
    """ END YOUR CODE HERE """
    return intersections


def get_support_mtx(intersections: np.ndarray, lines: np.ndarray, distance_threshold: float = DISTANCE_THRESHOLD):
    """
    Computes the I x L support matrix support_mtx where support_mtx[i, j] = 1 implies that the line lines[j] is close to
    the intersection intersections[j] and hence, supporting it.

    Args:
        intersections: I x 3 intersections [u, v, 1]
        lines: L x 3 lines [a, b, 1]
        distance_threshold: the maximum distance to be for the line to support the intersection.

    Returns:
        I x L support matrix
    """
    num_intersections, num_lines = intersections.shape[0], lines.shape[0]
    support_mtx = np.empty(
        shape=[num_intersections, num_lines], dtype=intersections.dtype)
    """ 
    YOUR CODE HERE
        1. Compute the I x L distance matrix distance_mtx between lines and intersections.
        2. Set the support matrix [i, j] to 1 if the distance_mtx[i,j] is below the distance threshold.
    """
    for i, intersection in enumerate(intersections):
        for j, line in enumerate(lines):
            numerator = abs((line[0]*intersection[0]) +
                            (line[1]*intersection[1]) + line[2])
            denominator = np.sqrt(line[0]**2 + line[1]**2)
            dist = float(numerator) / denominator
            if dist < distance_threshold:
                support_mtx[i][j] = 1
            else:
                support_mtx[i][j] = 0

    """ END YOUR CODE HERE """
    support_mtx = support_mtx.astype(int)
    assert support_mtx.shape == (num_intersections, num_lines)
    return support_mtx


def get_vanishing_pts(lines: np.ndarray, num_vanishing_pts: int) -> np.ndarray:
    """
    Computes the vanishing points using RANSAC.

    Args:
        lines: L x 3 numpy array of detected lines
        num_vanishing_pts: number of vansihing points to retrieve

    Returns:
        V x 3 array of vanishing points [u, v, 1]
    """
    intersections = get_pairwise_intersections(lines=lines)
    support_mtx = get_support_mtx(intersections=intersections, lines=lines)

    vanishing_pts = []
    """ 
    YOUR CODE HERE 
        - For num_vanishing_points, find the intersection with the most support as the vanishing point.
        - Then, remove its support set (of lines) from the list of lines.
        - Find the next vanishing points with the remaining lines (use a loop for this). 
        - Note: you don't have to recompute the support_mtx. You can set some entries to 0 to "remove" it.
    """
    for n in range(num_vanishing_pts):
        van_pt_idx = np.argmax(np.sum(a=support_mtx, axis=1))
        for idx, value in enumerate(support_mtx[van_pt_idx]):
            if value == 1:
                support_mtx[:, idx] = 0
        vanishing_pts.append(intersections[van_pt_idx])

    """ END YOUR CODE HERE """
    vanishing_pts = np.array(vanishing_pts)
    return vanishing_pts


def _save_vanishing_pts(save_prefix: os.path, vanishing_pts: np.ndarray):
    """ saves image of the vanishing points """
    image_file = '{}.jpg'.format(save_prefix)
    image = cv2.imread(image_file)

    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.plot(vanishing_pts[:, 0], vanishing_pts[:, 1],
             marker='o', color='r', ls='')
    save_dir = os.path.dirname(save_prefix)
    save_file = os.path.join(save_dir,
                             '{}-vanishing-pts.jpg'.format('horizontal' if 'nonvert' in save_prefix else 'vertical'))
    plt.savefig(save_file)


def get_vanishing_line(vanishing_pts: np.ndarray):
    """
    Returns the vanishing line from a pair of vanishing points

    Args:
        vanishing_pts: V x 3 array of vanishing points

    Returns:
        [a, b, 1] array of the vanishing line.
    """
    assert vanishing_pts.shape == (2, 3)
    vanishing_line = np.empty(shape=3, dtype=vanishing_pts.dtype)
    """ YOUR CODE HERE """
    vanishing_line = np.cross(vanishing_pts[0, :], vanishing_pts[-1, :])

    """ END YOUR CODE HERE """
    assert len(vanishing_line) == 3
    return vanishing_line


def _save_query_target_image(image: np.ndarray, query_info: dict, target_info: dict, save_dir: os.path):
    """ saves the image of the query and target height. """
    save_file = os.path.join(save_dir, 'query-target.jpg')
    save_image = copy.deepcopy(image)
    save_image = cv2.line(save_image, pt1=tuple(np.array(query_info['top']).astype(int)),
                          pt2=tuple(
                              np.array(query_info['bottom']).astype(int)),
                          color=(0, 255, 0), thickness=2)
    save_image = cv2.line(save_image,
                          pt1=tuple(np.array(target_info['top']).astype(int)),
                          pt2=tuple(
                              np.array(target_info['bottom']).astype(int)),
                          color=(0, 0, 255), thickness=2)
    cv2.imwrite(save_file, save_image)


def _parse_info_dict(info: dict):
    """ parses the info dict and returns the top and bottom coordinates of the object as [x,y,1]. """
    top, btm = np.ones(shape=3, dtype=float), np.ones(
        shape=3, dtype=float)
    top[:2] = info['top']
    btm[:2] = info['bottom']
    return top, btm


def get_target_height(vanishing_line: np.ndarray, query_info: dict, target_info: dict, vanishing_pt_v: np.ndarray):
    """
    Computes the target height from the query height, horizontal vanishing line, and vertical vanishing point

    Args:
        vanishing_line: the horizontal vanishing line [a, b, 1]
        query_info: the query info dictionary containing the top and bottom coords, and the query height
        target_info: the target info dictionary containing the top and bottom coords
        vanishing_pt_v: the vertical vanishing point [u, v, 1]

    Returns:

    """
    query_d1 = query_info['height']
    # top and bottom coordinates of the target
    pt_t2, pt_b2 = _parse_info_dict(info=target_info)
    # top and bottom coordinates of the query
    pt_t1, pt_b1 = _parse_info_dict(info=query_info)

    target_height = 0.
    """ 
    YOUR CODE HERE
    Compute the following: 
        1. vanishing point u
        2. transferred point
        3. distances
        4. distance ratio
        5. target height from distance ratio
    """
    def scale_coords(coords):
        return coords / coords[-1]

    vanishing_point_u = scale_coords(
        np.cross(np.cross(pt_b1, pt_b2), vanishing_line))
    transferred_pt = scale_coords(
        np.cross(np.cross(pt_t1, vanishing_point_u), np.cross(vanishing_pt_v, pt_b2)))
    dist_t1 = np.sqrt(
        (transferred_pt[0] - pt_b2[0])**2 + (transferred_pt[1] - pt_b2[1])**2)
    dist_t2 = np.sqrt((pt_t2[0] - pt_b2[0])**2+(pt_t2[1] - pt_b2[1])**2)
    dist_v = np.sqrt((vanishing_pt_v[0] - pt_b2[0])
                     ** 2 + (vanishing_pt_v[1] - pt_b2[1])**2)
    print(f"distv = {dist_v}  distt1 = {dist_t1}   distt2 = {dist_t2}")
    ratio = (dist_t1 * (dist_v - dist_t2)) / (dist_t2 * (dist_v - dist_t1))
    target_height = query_d1 / ratio

    """ END YOUR CODE HERE """
    return target_height


def _get_vert_nonvert_idxs(lines: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Returns the indexes of the vertical lines and the nonvertical lines from the """
    with np.errstate(divide='ignore'):
        vert_idxs = np.argwhere(np.bitwise_and(lines[:, 0] != 0,
                                               np.abs(lines[:, 1] / lines[:, 0]) < 3e-1)).reshape(-1)
    nonvert_idxs = np.setdiff1d(
        np.arange(lines.shape[0]), vert_idxs).reshape(-1)
    return nonvert_idxs, vert_idxs


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--img', type=str, default='P1080091', help='image name e.g. P1080091 corresponds to '
                                                                       'P1080091.jpg')
    argparser.add_argument('--nvp', type=int, default=3,
                           help='number of vanishing points')
    args = argparser.parse_args()

    save_dir = os.path.join(RESULT_DIR, args.img)
    os.makedirs(save_dir, exist_ok=True)
    image_file = os.path.join(DATA_DIR, '{}.jpg'.format(args.img))
    image = cv2.imread(image_file)

    # detect lines
    line_pts = detect_lines(image=image)
    _save_line_image(image=image, line_pts=line_pts,
                     save_prefix=os.path.join(save_dir, 'detected-lines'))
    lines = _get_lines_from_line_pts(line_pts=line_pts)

    # split lines into vertical and horizontal lines
    nonvert_idxs, vert_idxs = _get_vert_nonvert_idxs(lines=lines)
    vert_lines = lines[vert_idxs]
    vert_line_pts = line_pts[vert_idxs]
    nonvert_lines = lines[nonvert_idxs]
    nonvert_line_pts = line_pts[nonvert_idxs]
    _save_line_image(image=image, line_pts=nonvert_line_pts,
                     save_prefix=os.path.join(save_dir, 'detected-nonvert-lines'))
    _save_line_image(image=image, line_pts=vert_line_pts,
                     save_prefix=os.path.join(save_dir, 'detected-vert-lines'))

    # get horizontal and vertical vanishing points.
    horizontal_vanishing_pts = get_vanishing_pts(
        lines=nonvert_lines, num_vanishing_pts=2)
    vertical_vanishing_pt = get_vanishing_pts(
        lines=vert_lines, num_vanishing_pts=1)
    _save_vanishing_pts(save_prefix=os.path.join(save_dir, 'detected-vert-lines'),
                        vanishing_pts=vertical_vanishing_pt)
    _save_vanishing_pts(save_prefix=os.path.join(save_dir, 'detected-nonvert-lines'),
                        vanishing_pts=horizontal_vanishing_pts)
    vanishing_pts = np.concatenate(
        [horizontal_vanishing_pts, vertical_vanishing_pt], axis=0)
    np.save(os.path.join(save_dir, 'vanishing-pts.npy'), vanishing_pts)

    # load the query and target
    info_file = os.path.join(DATA_DIR, '{}-info.json'.format(args.img))
    with open(info_file, 'r') as f:
        info = json.load(f)
    query_info = info['query']
    target_info = info['target']
    _save_query_target_image(
        image=image, target_info=target_info, query_info=query_info, save_dir=save_dir)
    print('INFO: the query height is {}m'.format(query_info['height']))

    horizontal_vanishing_line = get_vanishing_line(
        vanishing_pts=horizontal_vanishing_pts)
    vert_vanishing_pt = vertical_vanishing_pt[0]
    target_height = get_target_height(vanishing_line=horizontal_vanishing_line, query_info=query_info,
                                      target_info=target_info,
                                      vanishing_pt_v=vert_vanishing_pt)
    np.save(os.path.join(save_dir, 'target-height.npy'), target_height)
    print('INFO: the target height is {:.2f}m'.format(target_height))


if __name__ == '__main__':
    main()
