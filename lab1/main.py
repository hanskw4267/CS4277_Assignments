"""
Name: Hans Kurnia
Matriculation number: A0184145E
"""

import os
import cv2
import numpy as np
import copy
from collections import Counter

DATASET = 'dunster'
NUM_INITIAL_MATCHES = 4

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PREDICTION_DIR = os.path.join(PROJECT_DIR, 'predictions')
DATASETS = ['']


def _load_dataset_images(dataset: str = DATASET):
    """ loads the dataset image """
    dataset_dir = os.path.join(DATA_DIR, dataset)
    img_fp1 = os.path.join(dataset_dir, '1.png')
    img_fp2 = os.path.join(dataset_dir, '2.png')

    img1 = cv2.imread(img_fp1)
    img2 = cv2.imread(img_fp2)
    return img1, img2


def get_keypoints(image: np.ndarray, num_keypoints: int = 250) -> (np.ndarray, np.ndarray):
    """
    Finds keypoints within image using SIFT.

    Args:
        image: numpy array of image
        num_keypoints: the number of retrieved keypoints
    Returns:
        N x 2 numpy array of keypoints represented by points (x,y).
    """

    keypoints, descriptors = np.array([]), np.array([])
    """
    YOUR CODE HERE: 
        - Call sift = cv2.SIFT_create with nfeatures=num_keypoints. 
        - Call sift.detectAndCompute with image=image and mask=None. This returns a list of keypoints 
    """
    sift = cv2.SIFT_create(nfeatures=num_keypoints)
    keypoints, descriptors = sift.detectAndCompute(image=image, mask=None)

    """ END YOUR CODE HERE """

    keypoints = np.array([list(kp.pt) for kp in keypoints])
    assert keypoints.shape[0] == descriptors.shape[0] == num_keypoints
    return keypoints, descriptors


def get_matches(descriptors1: np.ndarray, descriptors2: np.ndarray) -> list:
    """
    Returns list of cv2.Matches using feature descriptors sorted in ascending order of distance.
    See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

    Args:
        descriptors1: descriptors in the first image
        descriptors2: descriptors in the second image

    Returns:
        list of cv2.Matches
    """

    matches = []
    """ 
    YOUR CODE HERE:
        - Call bf = cv2.BFMatcher with crossCheck=True.
        - Call bf.match with descriptors1 and descriptors2 to obtain matches.
        - Sort matches in ascending order by their distance.
    """
    bf = cv2.BFMatcher_create(crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def _get_initial_keypoint_matches(image1: np.ndarray, image2: np.ndarray,
                                  num_matches: int = NUM_INITIAL_MATCHES) -> (np.ndarray, np.ndarray):
    """
    Finds the initial keypoint matches between the two images.

    Args:
        image1: numpy array of first image
        image2: numpy array of second image.
        num_matches: the number of top initial matches to retrieve, k

    Returns:
        k x 2 array of keypoints in the first image where Ni
        k x 2 array of keypoints in the second image
    """
    # get keypoints and descriptors
    keypoints1, descriptors1 = get_keypoints(image=image1)
    keypoints2, descriptors2 = get_keypoints(image=image2)

    # get the brute force matches from the descriptors.
    matches = get_matches(descriptors1, descriptors2)

    # add only unique keypoint matches. Reason: there can be duplicate matches found at different scales.
    selected_matches = []
    matches1, matches2 = [], []
    for match in matches:
        if len(selected_matches) >= num_matches:
            break
        i1, i2 = match.queryIdx, match.trainIdx
        keypoint1, keypoint2 = keypoints1[i1], keypoints2[i2]

        # check if the keypoint matches
        selected_match = '-'.join([str(keypoint1), str(keypoint2)])
        if selected_match in selected_matches:
            continue

        # add matched keypoint
        matches1.append(keypoint1)
        matches2.append(keypoint2)
        selected_matches.append(selected_match)
    matches1, matches2 = np.array(matches1), np.array(matches2)
    return matches1, matches2


def _read_line_pts(dataset: os.path, num_lines: int = 30) -> (np.ndarray, np.ndarray):
    """
    Reads the longest num_lines of line points from the dataset

    Args:
        dataset: the dataset to read.
        num_lines: the number of lines to read, L

    Returns:
        L x 4 numpy array of line points in the first image i.e. [x1, y1, x1', y1']
        L x 4 numpy array of line points in the second image i.e. [x2, y2, x2', y2']
    """
    dataset_dir = os.path.join(DATA_DIR, dataset)
    file1 = os.path.join(dataset_dir, 'ed1.txt')
    file2 = os.path.join(dataset_dir, 'ed2.txt')

    def _parse_line_file(_file: os.path) -> np.ndarray:
        with open(_file, 'r') as f:
            _line_pts = f.readlines()
        _lines = np.array([_line.strip().split()
                           for _line in _line_pts], dtype=float)
        _lengths = np.sqrt(np.sum((_lines[:, :2] - _lines[:, 2:])**2, axis=1))
        _idxs = np.argsort(_lengths, axis=0)[::-1]
        _lines = _lines[_idxs[:num_lines]]
        return _lines

    lines1 = _parse_line_file(_file=file1)
    lines2 = _parse_line_file(_file=file2)
    return lines1, lines2


def _save_image_detected_lines(image: np.ndarray, line_pts: np.ndarray, save_file: os.path,
                               line_thickness: float = 2, line_color: tuple = (0, 255, 0)):
    """ draw line_pts on image and saves the image into save_file. """
    assert line_pts.shape[1] == 4, 'should be N x 4 array'
    pts = copy.deepcopy(line_pts).astype(int)

    # draw lines
    save_image = copy.deepcopy(image)
    for x1, y1, x2, y2 in pts:
        cv2.line(save_image, (x1, y1), (x2, y2),
                 line_color, thickness=line_thickness)

    # save image
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    cv2.imwrite(save_file, save_image)


def _save_keypoint_matches(keypoints1: np.ndarray, keypoints2: np.ndarray, save_file: os.path):
    """
    Saves the keypoint matches between images as .npy file

    Args:
        keypoints1: N x 2 array of keypoints in the first image
        keypoints2: N x 2 array of keypoints in the second image e.g. keypoints1[i] corresponds to
                    keypoints2[i] N x 2 numpy array
        save_file: the numpy save file for to store the keypoints as N x 2 array
    """
    assert keypoints1.shape[1] == keypoints2.shape[1] == 2, 'assuming [u, v, 1], we only want [u, v]'
    assert keypoints1.shape[0] == keypoints2.shape[0]
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    matches = np.concatenate([keypoints1, keypoints2], axis=1)
    np.save(save_file, matches)


def _save_keypoint_match_image(image1: np.ndarray, image2: np.ndarray, keypoints1: np.ndarray, keypoints2: np.ndarray,
                               color: tuple, save_file: os.path):
    assert len(keypoints1) == len(keypoints2)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # concatenate the images along their widths.
    save_image = np.hstack([image1, image2])
    _keypoints2 = copy.deepcopy(keypoints2)
    _keypoints2[:, 0] += float(image1.shape[1])

    # draw the keypoint matches.
    for i, keypoint1 in enumerate(keypoints1):
        keypoint2 = _keypoints2[i]
        save_image = cv2.line(img=save_image, pt1=tuple(keypoint1[:2].astype(int)),
                              pt2=tuple(keypoint2[:2].astype(int)), thickness=2, color=color)
    cv2.imwrite(save_file, save_image)


def convert_line_pts_to_lines(pts: np.ndarray, other_pts: np.ndarray) -> np.ndarray:
    """
    Converts line points [u, v, u', v'] to line representation [a, b, 1] where au + bv + c = 0 for points
    [u, v, 1] that lies on the line.

    Args:
        pts: N x 3 numpy array of start points e.g. [u, v, 1]
        other_pts: N x 3 numpy array of other line pts e.g. [u', v', 1] pts[i] corresponds to other_pts[i] and they
        both form a line.

    Returns:
        N x 3 array of line representations [a, b, 1].
    """
    assert np.all(pts[:, -1] == 1) and np.all(other_pts[:, -1] == 1)
    assert pts.shape[1] == other_pts.shape[1] == 3, 'should be [u, v, 1] representation'

    lines = np.ones_like(pts)
    """ YOUR CODE HERE: hint: use np.cross """
    # Get line joining start/end points and scale to [a, b, 1] representation.
    for i in range(len(lines)):
        lines[i] = np.cross(pts[i], other_pts[i])
        lines[i] = lines[i] / lines[i][-1]
    """ END YOUR CODE HERE """
    assert np.all(lines[:, -1] == 1)
    return lines


def get_line_intersections(lines: np.ndarray, other_lines: np.ndarray) -> np.ndarray:
    """
    Computes the intersections between lines.

    Args:
        lines: N x 3 lines i.e. [a, b, 1]
        other_lines: N' x 3 lines i.e. [a', b', 1]

    Returns:
        N x 3 intersections [u, v, 1] where intersections[i] is the intersection between lines[i] and other_lines[i]
    """
    assert lines.shape[1] == other_lines.shape[1] == 3, 'should be [a, b, 1] representation'
    intersections = np.ones_like(lines)
    """ YOUR CODE HERE: hint: use np.cross """
    # Get line intersections and scale to [u, v, 1] representation.
    for i in range(len(lines)):
        intersections[i] = np.cross(lines[i], other_lines[i])
        intersections[i] = intersections[i] / intersections[i][-1]
    """ END YOUR CODE HERE """
    assert np.all(intersections[:, -1] ==
                  1), 'should be [u, v, 1] representation'
    assert intersections.shape[0] == lines.shape[0] or intersections.shape[0] == other_lines.shape[0]
    return intersections


def get_line_crossings(v_line: np.ndarray, r_lines: np.ndarray,
                       r_start_pts: np.ndarray, r_end_pts: np.ndarray) -> np.ndarray:
    """
    Return numpy array of line crossings in format [u, v, 1].

    Args:
        v_line: the virtual line [a, b, 1] as numpy array
        r_lines: L x 3 numpy array of detected (real) lines. i.e. [a, b, 1]
        r_start_pts: L x 3 numpy array of the start pts of the real lines i.e. [u, v, 1]
        r_end_pts: L x 3 numpy array of start end pts of the real lines i.e. [u', v', 1]

    Returns:
        [M, 3] numpy array of line crossings where M is the number of line crossings
    """

    assert v_line.shape[0] == r_lines.shape[1]
    assert len(v_line.shape) == 1 and len(r_lines.shape) == 2
    line_crossings = np.array([])  # dummy value.
    """ YOUR CODE HERE: hint: use get_line_intersections """
    # Make v_lines array of length L.
    v_lines = np.ones_like(r_lines)
    for i in range(len(v_lines)):
        v_lines[i] = v_line

    # Get line instersections of virtual line and real lines.
    line_intersections = get_line_intersections(v_lines, r_lines)

    # Check if intersection is between start/end points of real line segment.
    for i, pt in enumerate(line_intersections):
        prod = (pt[0] - r_start_pts[i][0]) * (r_end_pts[i][0] - r_start_pts[i][0]) + \
            (pt[1] - r_start_pts[i][1]) * (r_end_pts[i][1] - r_start_pts[i][1])
        sqrlen = (r_end_pts[i][0] - r_start_pts[i][0]) * (r_end_pts[i][0] - r_start_pts[i][0]) + \
            (r_end_pts[i][1] - r_start_pts[i][1]) * \
            (r_end_pts[i][1] - r_start_pts[i][1])
        if (prod >= 0 and prod <= sqrlen):
            line_crossings = np.concatenate((line_crossings, pt))

    # Reshpae line_crossings to be in shape [M, 3]
    line_crossings = np.reshape(line_crossings, (-1, 3))
    """ END YOUR CODE HERE """
    return line_crossings


def get_cross_ratios(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Compute cross ratios

    Args:
        a: N x 3 array of points A
        b: N x 3 array of points B
        c: N x 3 array of points C
        d: N x 3 array of points D

    Returns:
        N numpy array of cross_ratios where cross_ratios[i] corresponds to a[i], b[i], c[i], d[i].
    """
    assert a.shape == b.shape == c.shape == d.shape
    assert a.shape[1] == 3
    cross_ratios = np.ones(shape=a.shape[0], dtype=a.dtype)
    """ YOUR CODE HERE """

    # Function to calculate determinant, given 2 * [1, 2] points.
    def det(x, y):
        return (x[0] * y[1]) - (x[1] * y[0])

    # Calculate cross ratio of a,b,c,d
    for i in range(len(cross_ratios)):
        cross_ratios[i] = (det(a[i], b[i]) * det(c[i], d[i])) / \
            (det(a[i], c[i]) * det(b[i], d[i]))
    """ END YOUR CODE HERE """
    return cross_ratios


def _get_line_sweep_matches(v_start1: np.ndarray, v_end1: np.ndarray,
                            v_start2: np.ndarray, v_end2: np.ndarray,
                            line_crossings1: np.ndarray, line_crossings2: np.ndarray,
                            threshold: float = 5e-3) -> (np.ndarray, np.ndarray):
    """
    Performs single instance of line sweep for one virtual line pair and the set of line crossings.

    Args:
        v_start1: first keypoint of the virtual line in the first image
        v_end1: second keypoint of the virtual line in the first image
        v_start2: first keypoint of the virtual line in the second image
        v_end2: second keypoint of the virtual line in the second image
        line_crossings1: numpy array of line crossings in the first image
        line_crossings2: numpy array of line crossings in the second image
        threshold: threshold to consider cross ratios to be the same.

    Returns:
        O x 3 points in the first image
        O x 3 points in the second image, where out1[i] corresponds to out2[i]
    """
    def _get_cross_ratios(_start: np.ndarray, _end: np.ndarray, _crossings: np.ndarray) -> dict:
        _num_crossings = len(_crossings)
        _temp1 = np.concatenate([[i] * (_num_crossings - 1 - i)
                                 for i in range(_num_crossings - 1)], axis=0).reshape(-1)
        _temp2 = np.concatenate([list(range(i + 1, _num_crossings))
                                 for i in range(_num_crossings - 1)], axis=0).reshape(-1)

        _ci = np.concatenate([_temp1, _temp2], axis=0)
        _di = np.concatenate([_temp2, _temp1], axis=0)

        num_ratios = _ci.shape[0]
        _a = np.stack([_start] * num_ratios, axis=0)
        _b = np.stack([_end] * num_ratios, axis=0)
        _c = _crossings[_ci]
        _d = _crossings[_di]
        _cross_ratio = get_cross_ratios(a=_a, b=_b, c=_c, d=_d)
        _out = {'{}-{}'.format(_ci[_i], _di[_i]): _cross_ratio[_i]
                for _i in range(num_ratios)}
        return _out

    c_ratios1 = _get_cross_ratios(
        _start=v_start1, _end=v_end1, _crossings=line_crossings1)
    c_ratios2 = _get_cross_ratios(
        _start=v_start2, _end=v_end2, _crossings=line_crossings2)

    hypothesis = {i: [] for i in range(len(c_ratios1))}
    num_crossings1, num_crossings2 = len(line_crossings1), len(line_crossings2)
    for i1 in range(num_crossings1):
        for i2 in range(num_crossings2):
            for j1 in np.setdiff1d(range(num_crossings1), [i1]).reshape(-1):
                for j2 in np.setdiff1d(range(num_crossings2), [i2]).reshape(-1):
                    ratio1 = c_ratios1['{}-{}'.format(i1, j1)]
                    ratio2 = c_ratios2['{}-{}'.format(i2, j2)]
                    if np.abs(ratio1 - ratio2).item() <= threshold:
                        hypothesis[i1].append(i2)
                        hypothesis[j1].append(j2)

    num_hypos = np.array([len(v) for _, v in hypothesis.items()]).astype(int)
    hypo_idxs = np.argwhere(num_hypos > 0).reshape(-1)
    if len(hypo_idxs) == 0:
        return np.array([]), np.array([])

    pairwise_mtx = np.zeros([num_crossings1, num_crossings2])
    for i1, k in hypothesis.items():
        for i2, count in Counter(k).items():
            pairwise_mtx[i1, i2] = count

    out1, out2 = [], []
    for i1 in range(num_crossings1):
        i2 = np.argmax(pairwise_mtx[i1, :])
        if pairwise_mtx[i1, i2] == 0:
            continue
        out1.append(line_crossings1[i1, :])
        out2.append(line_crossings2[i2, :])
    out1, out2 = np.array(out1), np.array(out2)
    return out1, out2


def _get_new_keypoint_matches(keypoints1: np.ndarray, keypoints2: np.ndarray,
                              line_starts1: np.ndarray, line_ends1: np.ndarray,
                              line_starts2: np.ndarray, line_ends2: np.ndarray):
    """
    Gets complete set of new keypoint matches using the line sweep algorithm.

    Args:
        keypoints1: all initial keypoinst from SIFT matching in first image
        keypoints2: all initial keypoinst from SIFT matching in second image
        line_starts1: first line points of real lines in image1
        line_ends1: second line points of real lines in image1 i.e. line_starts1[i] corresponds to line_ends1[i]
        line_starts2: first line points of real lines in image2
        line_ends2: second line points of real lines in image2 i.e. line_starts2[i] corresponds to line_ends2[i]

    Returns:
        M x 3 new keypoints in the first image
        M x 3 new keypoints in the second image that matches to keypoints in first image.
    """
    assert keypoints1.shape == keypoints2.shape
    assert line_starts1.shape == line_ends1.shape
    assert line_starts2.shape == line_ends2.shape

    # get real lines
    r_lines1 = convert_line_pts_to_lines(
        pts=line_starts1, other_pts=line_ends1)
    r_lines2 = convert_line_pts_to_lines(
        pts=line_starts2, other_pts=line_ends2)

    # get virtual lines
    num_keypoints = len(keypoints1)
    v_start_i = np.concatenate([[i] * (num_keypoints-1-i)
                                for i in range(num_keypoints - 1)], axis=0).reshape(-1)
    v_end_i = np.concatenate([list(range(i+1, num_keypoints))
                              for i in range(num_keypoints - 1)], axis=0).reshape(-1)
    v_starts1, v_ends1 = keypoints1[v_start_i, :], keypoints1[v_end_i, :]
    v_starts2, v_ends2 = keypoints2[v_start_i, :], keypoints2[v_end_i, :]
    v_lines1 = convert_line_pts_to_lines(pts=v_starts1, other_pts=v_ends1)
    v_lines2 = convert_line_pts_to_lines(pts=v_starts2, other_pts=v_ends2)

    new_keypoints1, new_keypoints2 = [], []
    for i, v_line1 in enumerate(v_lines1):
        v_line2 = v_lines2[i]
        line_crossings1 = get_line_crossings(v_line=v_line1, r_lines=r_lines1,
                                             r_start_pts=line_starts1, r_end_pts=line_ends1)
        line_crossings2 = get_line_crossings(v_line=v_line2, r_lines=r_lines2,
                                             r_start_pts=line_starts2, r_end_pts=line_ends2)
        new_matches1, new_matches2 = _get_line_sweep_matches(v_start1=v_starts1[i], v_end1=v_ends1[i],
                                                             v_start2=v_starts2[i], v_end2=v_ends2[i],
                                                             line_crossings1=line_crossings1,
                                                             line_crossings2=line_crossings2)
        new_keypoints1.extend(list(new_matches1))
        new_keypoints2.extend(list(new_matches2))
    new_keypoints1, new_keypoints2 = np.array(
        new_keypoints1), np.array(new_keypoints2)
    assert len(new_keypoints1) == len(new_keypoints2)
    return new_keypoints1, new_keypoints2


def main():
    dataset = DATASET
    prediction_dir = os.path.join(PREDICTION_DIR, dataset)
    os.makedirs(prediction_dir, exist_ok=True)

    # detect keypoints
    image1, image2 = _load_dataset_images(dataset=dataset)
    keypoints1, keypoints2 = _get_initial_keypoint_matches(
        image1=image1, image2=image2)
    _save_keypoint_matches(keypoints1=keypoints1, keypoints2=keypoints2,
                           save_file=os.path.join(prediction_dir, 'initial-keypoint-matches.npy'))
    _save_keypoint_match_image(image1=image1, image2=image2, keypoints1=keypoints1, keypoints2=keypoints2,
                               color=(255, 0, 0),
                               save_file=os.path.join(prediction_dir, 'initial-keypoint-matches.png'))

    # read the longest detected lines
    line_pts1, line_pts2 = _read_line_pts(dataset=dataset)
    _save_image_detected_lines(image=image1, line_pts=line_pts1, save_file=os.path.join(prediction_dir,
                                                                                        'detected-line1.png'))
    _save_image_detected_lines(image=image2, line_pts=line_pts2, save_file=os.path.join(prediction_dir,
                                                                                        'detected-line2.png'))

    # get [u, v, 1] point representation
    keypoints1 = np.concatenate(
        [keypoints1, np.ones([keypoints1.shape[0], 1])], axis=1)
    keypoints2 = np.concatenate(
        [keypoints2, np.ones([keypoints2.shape[0], 1])], axis=1)

    # get [a, b, 1] line representation
    num_lines1, num_lines2 = line_pts1.shape[0], line_pts2.shape[0]
    start1, end1 = np.ones([num_lines1, 3], dtype=line_pts1.dtype), np.ones(
        [num_lines1, 3], dtype=line_pts1.dtype)
    start2, end2 = np.ones([num_lines2, 3], dtype=line_pts2.dtype), np.ones(
        [num_lines2, 3], dtype=line_pts2.dtype)
    start1[:, :2], end1[:, :2] = line_pts1[:, :2], line_pts1[:, 2:]
    start2[:, :2], end2[:, :2] = line_pts2[:, :2], line_pts2[:, 2:]

    # find new keypoint matches
    new_keypoints1, new_keypoints2 = _get_new_keypoint_matches(keypoints1=keypoints1,
                                                               keypoints2=keypoints2, line_starts1=start1,
                                                               line_ends1=end1, line_starts2=start2, line_ends2=end2)
    _save_keypoint_matches(keypoints1=new_keypoints1[:, :2], keypoints2=new_keypoints2[:, :2],
                           save_file=os.path.join(prediction_dir, 'new-keypoint-matches.npy'))
    _save_keypoint_match_image(image1=image1, image2=image2, keypoints1=new_keypoints1, keypoints2=new_keypoints2,
                               color=(0, 0, 255), save_file=os.path.join(prediction_dir, 'new-keypoint-matches.png'))

    final_keypoints1 = np.concatenate([keypoints1, new_keypoints1], axis=0)
    final_keypoints2 = np.concatenate([keypoints2, new_keypoints2], axis=0)
    _save_keypoint_match_image(image1=image1, image2=image2, keypoints1=final_keypoints1, keypoints2=final_keypoints2,
                               color=(255, 0, 0), save_file=os.path.join(prediction_dir, 'final-keypoint-matches.png'))


if __name__ == '__main__':
    main()
