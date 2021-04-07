""" CS4277/CS5477 Lab 5: STRUCTURE FROM MOTION AND BUNDLE ADJUSTMENT

Name: Hans Kurnia
Email: e0310940@u.nus.edu
Student ID: A0184145E
"""
import os
import numpy as np
import cv2

from tqdm import tqdm
import json
import open3d as o3d
import random
from scipy.optimize import least_squares

from preprocess import get_selected_points2d, get_camera_intrinsics
from preprocess import SCENE_GRAPH_FILE, RANSAC_MATCH_DIR, RANSAC_ESSENTIAL_DIR, HAS_BUNDLE_ADJUSTMENT, RESULT_DIR
from bundle_adjustment import compute_ba_residuals


def get_init_image_ids(scene_graph: dict) -> (str, str):
    """
    Returns the initial pair for incremental sfm. We pick the image pair with the largest number of inliers

    Args:
        scene_graph: dict of the scene graph where scene_graph[image_id1] returns the list of neighboring image ids
        of image_id1. scene graph is modelled like an adjacency list.

    Returns:
        [image_id1, image_id2]
    """
    max_pair = [None, None]  # dummy value
    """ YOUR CODE HERE """
    max_inliers = 0
    for u in scene_graph.keys():
        for v in scene_graph[u]:
            matches = load_matches(image_id1=u, image_id2=v)
            inliers = matches.shape[0]
            if inliers > max_inliers:
                max_pair = [u, v]
                max_inliers = inliers

    """ END YOUR CODE HERE """
    image_id1, image_id2 = sorted(max_pair)
    return image_id1, image_id2


def visualize_point_cloud(pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw([pcd])


def load_matches(image_id1: str, image_id2: str) -> np.ndarray:
    """ Returns N x 2 indexes of matches  [i,j] where keypoints1[i] at image_id1 corresponds to keypoints2[j]
    for image_id2 """
    sorted_nodes = sorted([image_id1, image_id2])
    match_id = '_'.join(sorted_nodes)
    match_file = os.path.join(RANSAC_MATCH_DIR, match_id + '.npy')
    matches = np.load(match_file)
    if sorted_nodes[0] == image_id2:
        matches = np.flip(matches, axis=1)
    return matches


def get_init_extrinsics(image_id1: str, image_id2: str, intrinsics: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Assume that the image_id1 is at [I|0] and second image_id2 is at [R|t] where R, t are derived from the
    essential matrix.

    Args:
        image_id1: first image id
        image_id2: second image id
        intrinsics: 3 x 3 camera intrinsics

    Returns:
        3 x 4 extrinsic matrix for image 1; 3 x 4 extrinsic matrix for image 2
    """
    extrinsics1 = np.zeros(shape=[3, 4], dtype=float)
    extrinsics1[:3, :3] = np.eye(3)

    match_id = '_'.join([image_id1, image_id2])
    essential_mtx_file = os.path.join(RANSAC_ESSENTIAL_DIR, match_id + '.npy')
    essential_mtx = np.load(essential_mtx_file)

    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    points2d_1 = get_selected_points2d(
        image_id=image_id1, select_idxs=matches[:, 0])
    points2d_2 = get_selected_points2d(
        image_id=image_id2, select_idxs=matches[:, 1])

    extrinsics2 = np.zeros(shape=[3, 4], dtype=float)
    """ YOUR CODE HERE """
    mask, R, t = None, None, None
    _, R, t, mask = cv2.recoverPose(essential_mtx, points2d_1,
                                    points2d_2, intrinsics, R, t, mask)
    extrinsics2 = np.hstack([R, t])

    """ END YOUR CODE HERE """
    return extrinsics1, extrinsics2


def initialize(scene_graph: dict, intrinsics: np.ndarray):
    """
    Performs initialization step.

    Args:
        scene_graph: dict of the scene graph where scene_graph[image_id1] returns the list of neighboring image ids
        of image_id1. scene graph is modelled like an adjacency list.
        intrinsics: 3x3 camera intrinsics

    Returns:
        image_id1: image at the world origin
        image_id2: neighbor of image_id1 where (image_id1, image_id2) has the highest number of RANSAC matches in
        the <scene_graph>
        extrinsics1: [I|0] extrinsic array
        extrinsics2: [R|t] extrinsic array from essential matrix
        points3d: points from triangulation betwene image_id1 and image_id2
        correspondences2d3d: dictionary of correspondences between 2d keypoints and 3d points for each image
            e.g. correspondences2d3d[image_id1][i] = j means that keypoint indexed at i in keypoint file for <image_id1>
            is correspondences to <points3d> indexed at j. Note that correspondences2d3d[image_id1] is a dictionary.
    """
    image_id1, image_id2 = get_init_image_ids(scene_graph)
    extrinsics1, extrinsics2 = get_init_extrinsics(
        image_id1=image_id1, image_id2=image_id2, intrinsics=intrinsics)
    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    points3d = triangulate(image_id1=image_id1, image_id2=image_id2, extrinsics1=extrinsics1,
                           extrinsics2=extrinsics2, intrinsics=intrinsics, kp_idxs1=matches[:, 0],
                           kp_idxs2=matches[:, 1])

    num_matches = matches.shape[0]
    correspondences2d3d = {
        image_id1: {matches[i, 0]: i for i in range(num_matches)},
        image_id2: {matches[i, 1]: i for i in range(num_matches)}
    }
    return image_id1, image_id2, extrinsics1, extrinsics2, points3d, correspondences2d3d


def triangulate(image_id1: str, image_id2: str, kp_idxs1: np.ndarray, kp_idxs2: np.ndarray,
                extrinsics1: np.ndarray, extrinsics2: np.ndarray, intrinsics: np.ndarray):
    proj_pts1 = get_selected_points2d(image_id=image_id1, select_idxs=kp_idxs1)
    proj_pts2 = get_selected_points2d(image_id=image_id2, select_idxs=kp_idxs2)

    proj_mtx1 = np.matmul(intrinsics, extrinsics1)
    proj_mtx2 = np.matmul(intrinsics, extrinsics2)

    points3d = cv2.triangulatePoints(projMatr1=proj_mtx1, projMatr2=proj_mtx2,
                                     projPoints1=proj_pts1.transpose(1, 0), projPoints2=proj_pts2.transpose(1, 0))
    points3d = points3d.transpose(1, 0)
    points3d = points3d[:, :3] / points3d[:, 3].reshape(-1, 1)
    return points3d


def get_reprojection_residuals(points2d: np.ndarray, points3d: np.ndarray, intrinsics: np.ndarray,
                               rotation_mtx: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Projects the 3d points back into the image and computes the residuals between each pair of points2d[i] and its
    corresponding reprojected point from points3d[i]

    Args:
        points2d: N x 2 array of 2d image points
        points3d: N x 3 arrya of 3d world points where points3d[i] coresponds to points2d[i] when reprojected.
        intrinsics: 3 x 3 camera intrinsic matrix
        rotation_mtx: 3 x 3 rotation matrix
        tvec: 3-dim rotation vector

    Returns:
        N array of residuals which is the euclidean distance between the points2d and their reprojected points.

    """
    residuals = np.zeros(points2d.shape[0])
    """ YOUR CODE HERE """
    def calc_dist(p1, p2):
        return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

    homography = intrinsics @ np.hstack([rotation_mtx, tvec.reshape(3, 1)])
    for i, pt_3d in enumerate(points3d):
        reprojected_pt2d = homography @ np.vstack(
            [pt_3d.reshape(3, 1), [1]])
        residuals[i] = calc_dist(
            points2d[i], reprojected_pt2d/reprojected_pt2d[-1])
    """ END YOUR CODE HERE """
    return residuals


def solve_pnp(image_id: str, point2d_idxs: np.ndarray, all_points3d: np.ndarray, point3d_idxs: np.ndarray,
              intrinsics: np.ndarray, num_ransac_iterations: int = 200, inlier_threshold: float = 5.0):
    """
    Solves the PnP problem using ransac.

    Args:
        image_id: id of the image
        point2d_idxs: indexes of image keypoints
        all_points3d: all 3d world points
        point3d_idxs: indexes of the <all_points3d> that correspond to the keypoints where
        all_points3d[point3d_idxs[i]] corresponds to keypoints[point2d_idxs[i]]
        intrinsics: 3 x 3 camera intrinsics
        num_ransac_iterations: number of ransac iterations
        inlier_threshold: threshold for residual where residual below threshold is an inlier.

    Returns:
        rotation matrix from pnp
        translation vector from pnp
        indexes of inlier matches
    """

    num_pts = point2d_idxs.shape[0]
    assert num_pts >= 6, 'there should be at least 6 points'

    points2d = get_selected_points2d(
        image_id=image_id, select_idxs=point2d_idxs)
    points3d = all_points3d[point3d_idxs]

    has_valid_solution = False
    max_rotation_mtx, max_tvec, max_is_inlier, max_num_inliers = None, None, None, 0
    for _ in range(num_ransac_iterations):
        selected_idxs = np.random.choice(
            num_pts, size=6, replace=False).reshape(-1)
        selected_pts2d = points2d[selected_idxs, :]
        selected_pts3d = points3d[selected_idxs, :]

        rotation_mtx, tvec = np.eye(3), np.zeros(
            3, dtype=float)  # dummy values
        residuals = np.zeros(shape=selected_pts2d.shape[0], dtype=float)
        """ 
        YOUR CODE HERE 
        1. call cv2.solvePnP(..., flags=cv2.SOLVEPNP_ITERATIVE, ...)
        2. convert the returned rotation vector to rotation matrix using cv2.Rodrigues
        3. compute the reprojection residuals
        """
        rvec = None
        _, rvec, tvec = cv2.solvePnP(
            selected_pts3d, selected_pts2d, intrinsics, distCoeffs=None, rvec=rvec, tvec=tvec, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=False)
        rotation_mtx, _ = cv2.Rodrigues(rvec, rotation_mtx)
        residuals = get_reprojection_residuals(
            points2d, points3d, intrinsics, rotation_mtx, tvec)

        """ END YOUR CODE HERE """

        is_inlier = residuals <= inlier_threshold
        num_inliers = np.sum(is_inlier).item()
        if num_inliers > max_num_inliers:
            max_rotation_mtx = rotation_mtx
            max_tvec = tvec
            max_is_inlier = is_inlier
            max_num_inliers = num_inliers
            has_valid_solution = True
    assert has_valid_solution
    inlier_idxs = np.argwhere(max_is_inlier).reshape(-1)
    return max_rotation_mtx, max_tvec, inlier_idxs


def add_points3d(image_id1: str, image_id2: str, all_extrinsic: dict, intrinsics, points3d: np.ndarray,
                 correspondences2d3d: dict):
    """
    From the image pair (image_id1, image_id2), triangulate to get new points3d. Update the correspondences
    for image_id1 and image_id2 and return the updated points3d as well.

    Args:
        image_id1: new image id
        image_id2: registered image id
        all_extrinsic: dictionary of image extrinsic where all_extrinsic[image_id1] returns the 3x4 extrinsic for
                        image_id1
        intrinsics: 3 x 3 camera intrinsic matrix
        points3d: M x 3 array of old 3d points
        correspondences2d3d: dictionary of correspondences between 2d keypoints and 3d points for each image
            e.g. correspondences2d3d[image_id1][i] = j means that keypoint indexed at i in keypoint file for <image_id1>
            is correspondences to <points3d> indexed at j. Note that correspondences2d3d[image_id1] is a dictionary.

    Returns:
        points3d: updated points3d
        correspondences2d3d: updated correspondences2d3d.
    """
    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    points2d_idxs2 = np.setdiff1d(
        matches[:, 1], correspondences2d3d[image_id2].keys()).reshape(-1)
    if len(points2d_idxs2) == 0:
        return points3d, correspondences2d3d  # no new registration

    # triangulate new points that were not registered
    matches_idxs = np.array(
        [np.argwhere(matches[:, 1] == i).reshape(-1)[0] for i in points2d_idxs2])
    matches = matches[matches_idxs, :]
    """ 
    START YOUR CODE HERE:
    triangulate between the image points for the unregistered matches for image_id1 and image_id2 to get new points3d
    new_points3d = triangulate(..., kp_idxs1=matches[:, 0], kp_idxs2=matches[:, 1], ...)
    """
    new_points3d = triangulate(image_id1, image_id2, kp_idxs1=matches[:, 0], kp_idxs2=matches[:, 1],
                               extrinsics1=all_extrinsic[image_id1], extrinsics2=all_extrinsic[image_id2], intrinsics=intrinsics)

    """ END YOUR CODE HERE """

    num_new_points3d = new_points3d.shape[0]
    new_points3d_idxs = np.arange(num_new_points3d) + points3d.shape[0]
    correspondences2d3d[image_id1] = {
        matches[i, 0]: new_points3d_idxs[i] for i in range(num_new_points3d)}
    for i in range(num_new_points3d):
        correspondences2d3d[image_id1][matches[i, 0]] = new_points3d_idxs[i]
        correspondences2d3d[image_id2][matches[i, 1]] = new_points3d_idxs[i]
    points3d = np.concatenate([points3d, new_points3d], axis=0)
    return points3d, correspondences2d3d


def get_next_pair(scene_graph: dict, registered_ids: list):
    """
    Finds the next match where the one of the images is unregistered while the other is registered. The next image pair
    is the one that has highest number of inliers.

    Args:
        scene_graph: dict of the scene graph where scene_graph[image_id1] returns the list of neighboring image ids
        of image_id1. scene graph is modelled like an adjacency list.
        registered_ids: list of registered image ids

    Returns:
        new_id: new image id to be registered
        registered_id: registered image id that has highest number of inliers along with the new_id
    """
    max_new_id, max_registered_id, max_num_inliers = None, None, 0
    """ YOUR CODE HERE """
    for u in registered_ids:
        for v in [x for x in scene_graph[u] if x not in registered_ids]:
            matches = load_matches(image_id1=u, image_id2=v)
            inliers = matches.shape[0]
            if inliers > max_num_inliers:
                max_registered_id, max_new_id = u, v
                max_num_inliers = inliers
    """ END YOUR CODE HERE """
    return max_new_id, max_registered_id


def get_pnp_2d3d_correspondences(image_id1: str, image_id2: str, correspondences2d3d: dict) -> (np.ndarray, np.ndarray):
    """
    Returns 2d and 3d correspondences for the image_id1 and the current world points. We use the transitive property
    where matches[i, 0] -> matches[i, 1] -> correspondences[image_id2][matches[i,1]], where image_id2 is a registered
    image

    Args:
        image_id1: new image id
        image_id2: registered image id
        correspondences2d3d: dictionary of correspondences

    Returns:
        points2d_idxs: keypoint idxs of first image
        points3d_idxs: points3d idxs
    """
    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    points2d_idxs2 = np.intersect1d(matches[:, 1], list(
        correspondences2d3d[image_id2].keys())).reshape(-1)
    match_idxs = [np.argwhere(matches[:, 1] == i).reshape(-1)[0]
                  for i in points2d_idxs2]
    match_idxs = np.array(match_idxs)
    points2d_idxs1 = matches[match_idxs, 0]
    point3d_idxs = np.array([correspondences2d3d[image_id2][i]
                             for i in points2d_idxs2])
    return points2d_idxs1, point3d_idxs


def bundle_adjustment(registered_ids: list, points3d: np.ndarray, correspondences2d3d: np.ndarray,
                      all_extrinsics: dict, intrinsics: np.ndarray, max_nfev: int = 30):
    # create parameters
    parameters = []
    for image_id in registered_ids:
        # convert rotation matrix to Rodriguez vector
        extrinsics = all_extrinsics[image_id]
        rotation_mtx = extrinsics[:3, :3]
        tvec = extrinsics[:, 3].reshape(3)
        rvec, _ = cv2.Rodrigues(rotation_mtx)
        rvec = rvec.reshape(3)

        parameters.append(rvec)
        parameters.append(tvec)
    parameters.append(points3d.reshape(-1))
    parameters = np.concatenate(parameters, axis=0)

    # create correspondences
    points2d, camera_idxs, points3d_idxs = [], [], []
    for i, image_id in enumerate(registered_ids):
        correspondence_dict = correspondences2d3d[image_id]
        correspondences = np.array([[k, v]
                                    for k, v in correspondence_dict.items()])
        pt2d_idxs = correspondences[:, 0]
        pt3d_idxs = correspondences[:, 1]

        pt2d = get_selected_points2d(image_id=image_id, select_idxs=pt2d_idxs)
        points2d.append(pt2d)
        points3d_idxs.append(pt3d_idxs)
        camera_idxs.append(np.ones(pt2d.shape[0]) * i)

    num_cameras = len(registered_ids)
    points2d = np.concatenate(points2d, axis=0)
    camera_idxs = np.concatenate(camera_idxs, axis=0).astype(int)
    points3d_idxs = np.concatenate(points3d_idxs, axis=0).astype(int)

    # run optimization
    results = least_squares(fun=compute_ba_residuals, x0=parameters, method='trf', max_nfev=max_nfev,
                            args=(intrinsics, num_cameras, points2d, camera_idxs, points3d_idxs), verbose=2)

    updated_parameters = results.x
    camera_parameters = updated_parameters[:num_cameras * 6]
    camera_parameters = camera_parameters.reshape(num_cameras, 6)
    for i, image_id in enumerate(registered_ids):
        params = camera_parameters[i]
        rvec, tvec = params[:3], params[3:]
        rvec = rvec.reshape(1, 3)
        rotation_mtx, _ = cv2.Rodrigues(rvec)
        extrinsics = np.concatenate(
            [rotation_mtx, tvec.reshape(-1, 1)], axis=1)
        all_extrinsics[image_id] = extrinsics
    points3d = updated_parameters[num_cameras * 6:].reshape(-1, 3)
    return all_extrinsics, points3d


def incremental_sfm(registered_ids: list, all_extrinsic: dict, intrinsics: np.ndarray, points3d: np.ndarray,
                    correspondences2d3d: dict, scene_graph: dict, has_bundle_adjustment: bool) -> \
        (np.ndarray, dict, dict, list):
    all_image_ids = list(scene_graph.keys())
    num_steps = len(all_image_ids) - 2
    for _ in tqdm(range(num_steps)):
        # get pose for new image
        new_id, registered_id = get_next_pair(
            scene_graph=scene_graph, registered_ids=registered_ids)
        points2d_idxs1, points3d_idxs = get_pnp_2d3d_correspondences(image_id1=new_id, image_id2=registered_id,
                                                                     correspondences2d3d=correspondences2d3d)
        rotation_mtx, tvec, inlier_idxs = solve_pnp(image_id=new_id, point2d_idxs=points2d_idxs1,
                                                    all_points3d=points3d, point3d_idxs=points3d_idxs,
                                                    intrinsics=intrinsics)

        # update correspondences
        new_extrinsics = np.concatenate(
            [rotation_mtx, tvec.reshape(-1, 1)], axis=1)
        all_extrinsic[new_id] = new_extrinsics
        correspondences2d3d[new_id] = {
            points2d_idxs1[i]: points3d_idxs[i] for i in inlier_idxs}

        # create new points + update correspondences
        points3d, correspondences2d3d = add_points3d(image_id1=new_id, image_id2=registered_id,
                                                     all_extrinsic=all_extrinsic,
                                                     intrinsics=intrinsics, points3d=points3d,
                                                     correspondences2d3d=correspondences2d3d)
        registered_ids.append(new_id)

    if has_bundle_adjustment:
        all_extrinsic, points3d = bundle_adjustment(registered_ids=registered_ids, points3d=points3d,
                                                    all_extrinsics=all_extrinsic, intrinsics=intrinsics,
                                                    correspondences2d3d=correspondences2d3d)
    assert len(np.setdiff1d(all_image_ids, registered_ids).reshape(-1)) == 0
    return points3d, all_extrinsic, correspondences2d3d, registered_ids


def main():
    # set seeds
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)

    with open(SCENE_GRAPH_FILE, 'r') as f:
        scene_graph = json.load(f)

    # run initialization step
    intrinsics = get_camera_intrinsics()
    image_id1, image_id2, extrinsic1, extrinsic2, points3d, correspondences2d3d = \
        initialize(scene_graph=scene_graph, intrinsics=intrinsics)
    registered_ids = [image_id1, image_id2]
    all_extrinsic = {
        image_id1: extrinsic1,
        image_id2: extrinsic2
    }

    points3d, all_extrinsic, correspondences2d3d, registered_ids = \
        incremental_sfm(registered_ids=registered_ids, all_extrinsic=all_extrinsic, intrinsics=intrinsics,
                        correspondences2d3d=correspondences2d3d, points3d=points3d, scene_graph=scene_graph,
                        has_bundle_adjustment=HAS_BUNDLE_ADJUSTMENT)

    os.makedirs(RESULT_DIR, exist_ok=True)
    points3d_save_file = os.path.join(RESULT_DIR, 'points3d.npy')
    np.save(points3d_save_file, points3d)

    correspondences2d3d = {a: {int(c): int(d) for c, d in b.items()}
                           for a, b in correspondences2d3d.items()}
    correspondences2d3d_save_file = os.path.join(
        RESULT_DIR, 'correspondences2d3d.json')
    with open(correspondences2d3d_save_file, 'w') as f:
        json.dump(correspondences2d3d, f, indent=1)

    all_extrinsic = {image_id: [list(row) for row in extrinsic.astype(float)]
                     for image_id, extrinsic in all_extrinsic.items()}
    extrinsic_save_file = os.path.join(RESULT_DIR, 'all-extrinsic.json')
    with open(extrinsic_save_file, 'w') as f:
        json.dump(all_extrinsic, f, indent=1)

    registration_save_file = os.path.join(
        RESULT_DIR, 'registration-trajectory.txt')
    with open(registration_save_file, 'w') as f:
        for image_id in registered_ids:
            f.write(image_id + '\n')


if __name__ == '__main__':
    main()
