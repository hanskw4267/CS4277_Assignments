""" CS4277/CS5477 Lab 5: STRUCTURE FROM MOTION AND BUNDLE ADJUSTMENT

Name: Hans Kurnia
Email: e0310940@u.nus.edu
Student ID: A0184145E
"""
import os
import numpy as np
import cv2
import pickle as pkl
from tqdm import tqdm
import networkx as nx
import shutil
import json
import torch.utils.data as tdata
import argparse

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
PREDICTION_DIR = os.path.join(PROJECT_DIR, 'predictions')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str,
                       choices=['temple', 'mini-temple'])
argparser.add_argument('--ba', action='store_true')
args = argparser.parse_args()

DATASET = args.dataset
DATASET_DIR = os.path.join(DATA_DIR, DATASET)
IMAGE_DIR = os.path.join(DATASET_DIR, 'images')
INTRINSICS_FILE = os.path.join(DATASET_DIR, 'intrinsics.txt')

SAVE_DIR = os.path.join(PREDICTION_DIR, DATASET)
BAD_MATCHES_FILE = os.path.join(SAVE_DIR, 'bad-match.txt')
KEYPOINT_DIR = os.path.join(SAVE_DIR, 'keypoints')
BF_MATCH_DIR = os.path.join(SAVE_DIR, 'bf-match')
BF_MATCH_IMAGE_DIR = os.path.join(SAVE_DIR, 'bf-match-images')

RANSAC_MATCH_DIR = os.path.join(SAVE_DIR, 'ransac-match')
RANSAC_ESSENTIAL_DIR = os.path.join(SAVE_DIR, 'ransac-fundamental')
RANSAC_MATCH_IMAGE_DIR = os.path.join(SAVE_DIR, 'ransac-match-images')
BAD_RANSAC_MATCHES_FILE = os.path.join(SAVE_DIR, 'bad-ransac-matches.txt')
SCENE_GRAPH_FILE = os.path.join(SAVE_DIR, 'scene-graph.json')

HAS_BUNDLE_ADJUSTMENT = args.ba
SPLIT = 'bundle-adjustment' if HAS_BUNDLE_ADJUSTMENT else 'no-bundle-adjustment'
RESULT_DIR = os.path.join(SAVE_DIR, 'results', SPLIT)

assert not (HAS_BUNDLE_ADJUSTMENT and DATASET == 'temple'), \
    'fail safe for students; remove line if u have the the resources to do BA for large cases and interested.'


class ParallelDataset(tdata.Dataset):
    def __init__(self, data: list, func):
        """
        Args:
            data: list of tuples of data points
            func: function to run for each data point
        """
        super(ParallelDataset, self).__init__()
        self.data = data
        self.func = func

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        out = self.func(*data)
        return out


def get_camera_intrinsics() -> np.ndarray:
    """ loads the camera intrinsics and return it as 3x3 intrinsic camera matrix """
    with open(INTRINSICS_FILE, 'r') as f:
        intrinsics = f.readlines()
    intrinsics = [line.strip().split(' ') for line in intrinsics]
    intrinsics = np.array(intrinsics).astype(float)
    return intrinsics


def encode_keypoint(kp: cv2.KeyPoint) -> tuple:
    """ encodes keypoint into a tuple for saving """
    return kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id


def decode_keypoint(kp: tuple) -> cv2.KeyPoint:
    """ decodes keypoint back into cv2.KeyPoint class. """
    return cv2.KeyPoint(x=kp[0][0], y=kp[0][1], _size=kp[1], _angle=kp[2], _response=kp[3],
                        _octave=kp[4], _class_id=kp[5])


def get_detected_keypoints(image_id: str) -> (list, list):
    """ Returns detected list of cv2.KeyPoint and their corresponding descriptors. """
    keypoint_file = os.path.join(KEYPOINT_DIR, image_id + '.pkl')
    with open(keypoint_file, 'rb') as _f:
        keypoint = pkl.load(_f)
    keypoints, descriptors = keypoint['keypoints'], keypoint['descriptors']
    keypoints = [decode_keypoint(_kp) for _kp in keypoints]
    return keypoints, descriptors


def parallel_processing(data: list, func, batchsize: int = 1, shuffle: bool = False, num_workers: int = 6):
    """ code to run preprocessing functions in parallel. """
    dataset = ParallelDataset(data=data, func=func)
    dataloader = tdata.DataLoader(
        dataset=dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batchsize)
    out = []
    for batch_out in tqdm(dataloader):
        out.extend(list(batch_out))
    return out


def detect_keypoints(image_file: os.path):
    """
    Detects SIFT keypoints in <image_file> and store it the detected keypoints into a pickle file. Returns the image_id

    Args:
        image_file: path to image file.
    """

    image_id = os.path.basename(image_file)[:-4]
    save_file = os.path.join(KEYPOINT_DIR, image_id + '.pkl')

    keypoints, descriptors = [], []
    """ YOUR CODE HERE:
    Detect keypoints using cv2.SIFT_create() and sift.detectAndCompute
    """
    img = cv2.imread(image_file)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image=img, mask=None)

    """ END YOUR CODE HERE. """

    keypoints = [encode_keypoint(kp=kp) for kp in keypoints]
    save_dict = {
        'keypoints': keypoints,
        'descriptors': descriptors
    }

    with open(save_file, 'wb') as f:
        pkl.dump(save_dict, f)
    return image_id


def create_feature_matches(image_file1: os.path, image_file2: os.path, lowe_ratio: float = 0.6, min_matches: int = 10):
    """
    1. Match the detected keypoint features between the two images in <image_file1> and <image_file2> using the
        descriptors. There would be two possible matches for each keypoint.
    2. Use the Lowe Ratio test to see to filter out noisy matches i.e. the second possible point is also a good match
        relative to the first point. See https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html for
        similar implementation.
    3. The feature matches are saved as an N x 2 numpy array of indexes [i,j] where keypoints1[i] is matched with
        keypoints2[j]

    Args:
        image_file1: path to first image file
        image_file2: path to second image file
        lowe_ratio: the ratio for the Lowe ratio test. Good matches are when the first match has distance less than the
        <lowe_ratio> x distance of the second best match.
        min_matches: the minimum number of matches for the feature matches to exist..
    """
    image_id1 = os.path.basename(image_file1)[:-4]
    image_id2 = os.path.basename(image_file2)[:-4]
    match_id = '{}_{}'.format(image_id1, image_id2)

    match_save_file = os.path.join(BF_MATCH_DIR, match_id + '.npy')
    image_save_file = os.path.join(BF_MATCH_IMAGE_DIR, match_id + '.png')

    keypoints1, descriptors1 = get_detected_keypoints(image_id=image_id1)
    keypoints2, descriptors2 = get_detected_keypoints(image_id=image_id2)

    good_matches = []
    """ 
    YOUR CODE HERE: 
    1. Run cv.BFMatcher() and matcher.knnMatch(descriptors1, descriptors2, 2)
    2. Filter the feature matches using the Lowe ratio test.
    """
    bf = cv2.BFMatcher_create()
    matches = bf.knnMatch(descriptors1, descriptors2, 2)

    for a, b in matches:
        if a.distance < lowe_ratio * b.distance:
            good_matches.append([a])

    """ END YOUR CODE HERE. """
    if len(good_matches) < min_matches:
        return match_id

    # image visualization of feature matching
    image1 = cv2.imread(image_file1)
    image2 = cv2.imread(image_file2)
    save_image = cv2.drawMatchesKnn(img1=image1, keypoints1=keypoints1, img2=image2, keypoints2=keypoints2,
                                    matches1to2=good_matches, outImg=None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(image_save_file, save_image)
    good_matches = [match[0] for match in good_matches]

    # save the feature matches
    feature_matches = []
    for match in good_matches:
        feature_matches.append([match.queryIdx, match.trainIdx])
    feature_matches = np.array(feature_matches)
    np.save(match_save_file, feature_matches)
    return match_id


def get_selected_points2d(image_id: str, select_idxs: np.ndarray) -> np.ndarray:
    """ loaded selected keypoint 2d coordinates from <select_idxs> """
    keypoints, _ = get_detected_keypoints(image_id=image_id)
    points2d = [keypoints[i].pt for i in select_idxs]
    points2d = np.array(points2d)
    return points2d


def create_ransac_matches(image_file1: os.path, image_file2: os.path,
                          min_feature_matches: int = 30, ransac_threshold: float = 1.0):
    """
    Performs geometric verification of feature matches using RANSAC. We will remove image matches that have less
    than <min_num_inliers> number of geometrically-verified matches.

    Args:
        image_file1: path to the first image file
        image_file2: path to the second image file
        min_feature_matches: minimum number of feature matches to qualify as inputs
        ransac_threshold: the reprojection error threshold for RANSAC

    Returns:
        the match id i.e. <image_id1, image_id2>
    """
    image_id1 = os.path.basename(image_file1)[:-4]
    image_id2 = os.path.basename(image_file2)[:-4]
    match_id = '{}_{}'.format(image_id1, image_id2)

    match_save_file = os.path.join(RANSAC_MATCH_DIR, match_id + '.npy')
    essential_mtx_save_file = os.path.join(
        RANSAC_ESSENTIAL_DIR, match_id + '.npy')
    image_save_file = os.path.join(RANSAC_MATCH_IMAGE_DIR, match_id + '.png')
    feature_match_file = os.path.join(BF_MATCH_DIR, match_id + '.npy')

    if not os.path.exists(feature_match_file):
        return match_id  # images are not matched under feature matching or the image_id1 >= image_id2

    match_idxs = np.load(feature_match_file)
    if match_idxs.shape[0] < min_feature_matches:
        return match_id  # there are too little inliers

    points1 = get_selected_points2d(
        image_id=image_id1, select_idxs=match_idxs[:, 0])
    points2 = get_selected_points2d(
        image_id=image_id2, select_idxs=match_idxs[:, 1])
    camera_intrinsics = get_camera_intrinsics()

    is_inlier = np.ones(shape=points1.shape[0], dtype=bool)  # dummy value
    essential_mtx = np.zeros(shape=[3, 3], dtype=float)
    """ 
    YOUR CODE HERE 
    Perform goemetric verification by finding the essential matrix between keypoints in the first image and keypoints in
    the second image using cv2.findEssentialMatrix(..., method=cv2.RANSAC, threshold=ransac_threshold, ...)
    """
    essential_mtx, is_inlier = cv2.findEssentialMat(
        points1, points2, camera_intrinsics, cv2.RANSAC, 0.999, ransac_threshold)

    """ END YOUR CODE HERE """

    is_inlier = is_inlier.ravel().tolist()
    inlier_idxs = np.argwhere(is_inlier).reshape(-1)
    if len(inlier_idxs) == 0:
        return match_id

    inlier_idxs = match_idxs[inlier_idxs, :]
    np.save(match_save_file, inlier_idxs)
    np.save(essential_mtx_save_file, essential_mtx)

    # save visualization image
    image1 = cv2.imread(image_file1)
    image2 = cv2.imread(image_file2)
    save_image = np.concatenate([image1, image2], axis=1)
    offset = image1.shape[1]
    match_pts = np.concatenate([points1, points2], axis=1)
    match_pts = match_pts.astype(int)
    for x1, y1, x2, y2 in match_pts:
        save_image = cv2.line(img=save_image, pt1=(x1, y1), pt2=(
            x2 + offset, y2), thickness=1, color=(0, 255, 0))
    cv2.imwrite(image_save_file, save_image)
    return match_id


def create_scene_graph(image_files: list, min_num_inliers: int = 40):
    graph = nx.Graph()
    graph.add_nodes_from(list(range(len(image_files))))
    image_ids = [os.path.basename(file)[:-4] for file in image_files]
    """ 
    YOUR CODE HERE:
    Add edges to <graph> if the minimum number of geometrically verified inliers between images is at least  
    <min_num_inliers> 
    """
    for i, u in enumerate(image_ids):
        for j, v in enumerate(image_ids):
            if (u != v):
                fn = os.path.join(RANSAC_MATCH_DIR, f"{u}_{v}.npy")
                if (os.path.exists(fn)):
                    if len(np.load(fn)) >= min_num_inliers:
                        graph.add_edge(i, j)
    """ END YOUR CODE HERE """

    graph_dict = {node: [] for node in image_ids}
    for i1, i2 in graph.edges:
        node1 = image_ids[i1]
        node2 = image_ids[i2]
        graph_dict[node1].append(node2)
        graph_dict[node2].append(node1)
    graph_dict = {node: list(np.unique(neighbors).reshape(-1))
                  for node, neighbors in graph_dict.items()}
    with open(SCENE_GRAPH_FILE, 'w') as f:
        json.dump(graph_dict, f, indent=1)


def preprocess(image_files: list):
    print('INFO: detecting image keypoints...')
    shutil.rmtree(KEYPOINT_DIR, ignore_errors=True)
    os.makedirs(KEYPOINT_DIR, exist_ok=True)
    parallel_processing(data=[(file,)
                              for file in image_files], func=detect_keypoints)

    matches = []
    for i, file1 in enumerate(image_files):
        for file2 in image_files[i+1:]:
            matches.append((file1, file2))
    shutil.rmtree(BF_MATCH_DIR, ignore_errors=True)
    shutil.rmtree(BF_MATCH_IMAGE_DIR, ignore_errors=True)
    os.makedirs(BF_MATCH_DIR, exist_ok=True)
    os.makedirs(BF_MATCH_IMAGE_DIR, exist_ok=True)
    print('INFO: creating pairwise matches between images...')
    parallel_processing(data=matches, func=create_feature_matches)

    print('INFO: creating ransac matches...')
    shutil.rmtree(RANSAC_MATCH_DIR, ignore_errors=True)
    shutil.rmtree(RANSAC_MATCH_IMAGE_DIR, ignore_errors=True)
    shutil.rmtree(RANSAC_ESSENTIAL_DIR, ignore_errors=True)
    os.makedirs(RANSAC_MATCH_DIR, exist_ok=True)
    os.makedirs(RANSAC_MATCH_IMAGE_DIR, exist_ok=True)
    os.makedirs(RANSAC_ESSENTIAL_DIR, exist_ok=True)
    parallel_processing(data=matches, func=create_ransac_matches)

    print('INFO: creating scene graph...')
    create_scene_graph(image_files=image_files)


def main():
    image_files = [os.path.join(IMAGE_DIR, filename)
                   for filename in sorted(os.listdir(IMAGE_DIR))]

    print('INFO: detecting image keypoints...')
    shutil.rmtree(KEYPOINT_DIR, ignore_errors=True)
    os.makedirs(KEYPOINT_DIR, exist_ok=True)
    parallel_processing(data=[(file,)
                              for file in image_files], func=detect_keypoints)

    print('INFO: creating pairwise matches between images...')
    matches = []
    for i, file1 in enumerate(image_files):
        for file2 in image_files[i + 1:]:
            matches.append((file1, file2))
    shutil.rmtree(BF_MATCH_DIR, ignore_errors=True)
    shutil.rmtree(BF_MATCH_IMAGE_DIR, ignore_errors=True)
    os.makedirs(BF_MATCH_DIR, exist_ok=True)
    os.makedirs(BF_MATCH_IMAGE_DIR, exist_ok=True)
    parallel_processing(data=matches, func=create_feature_matches)

    print('INFO: creating ransac matches...')
    shutil.rmtree(RANSAC_MATCH_DIR, ignore_errors=True)
    shutil.rmtree(RANSAC_MATCH_IMAGE_DIR, ignore_errors=True)
    shutil.rmtree(RANSAC_ESSENTIAL_DIR, ignore_errors=True)
    os.makedirs(RANSAC_MATCH_DIR, exist_ok=True)
    os.makedirs(RANSAC_MATCH_IMAGE_DIR, exist_ok=True)
    os.makedirs(RANSAC_ESSENTIAL_DIR, exist_ok=True)
    parallel_processing(data=matches, func=create_ransac_matches)

    print('INFO: creating scene graph...')
    create_scene_graph(image_files=image_files)


if __name__ == '__main__':
    main()
