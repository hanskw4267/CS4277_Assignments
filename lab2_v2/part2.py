import os
from main import *
from helper import *


STR_OUTPUT_MISMATCH = 'Errors in'

DATASET = 'inputs'
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')
DATASETS = ['']


def warp_images_all(images, h_matrices):
    """Warps all images onto a black canvas.

    Note: We implemented this function for you, but it'll be useful to note
     the necessary steps
     1. Compute the bounds of each of the images (which can be negative)
     2. Computes the necessary size of the canvas
     3. Adjust all the homography matrices to the canvas bounds
     4. Warp images

    Requires:
        transform_homography(), warp_image()

    Args:
        images (List[np.ndarray]): List of images to warp
        h_matrices (List[np.ndarray]): List of homography matrices

    Returns:
        stitched (np.ndarray): Stitched images
    """
    assert len(images) == len(h_matrices) and len(images) > 0
    num_images = len(images)

    corners_transformed = []
    for i in range(num_images):
        h, w = images[i].shape[:2]
        bounds = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
        transformed_bounds = transform_homography(bounds, h_matrices[i])
        corners_transformed.append(transformed_bounds)
    corners_transformed = np.concatenate(corners_transformed, axis=0)

    # Compute required canvas size
    min_x, min_y = np.min(corners_transformed, axis=0)
    max_x, max_y = np.max(corners_transformed, axis=0)
    min_x, min_y = floor(min_x), floor(min_y)
    max_x, max_y = ceil(max_x), ceil(max_y)

    canvas = np.zeros((max_y-min_y, max_x-min_x, 3), images[0].dtype)

    for i in range(num_images):
        # adjust homography matrices
        trans_mat = np.array([[1.0, 0.0, -min_x],
                              [0.0, 1.0, -min_y],
                              [0.0, 0.0, 1.0]], h_matrices[i].dtype)
        h_adjusted = trans_mat @ h_matrices[i]

        # Warp
        canvas = warp_image(images[i], canvas, h_adjusted)

    return canvas


def compute_homography_error_test():
    """
    Implement the following function(s): compute_homography_error()
    """
    # test case 1
    src1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    dst1 = np.array([[1.0, 2.0], [1.5, 2.5], [2.5, 3]])
    homo1 = np.array([[0.5, 0.0, 1.0],
                      [0.0, 0.5, 2.0],
                      [0.0, 0.0, 1.0]])
    error1 = compute_homography_error(src1, dst1, homo1)
    # test case 2
    src2 = np.array([[434.375, 93.625], [429.625, 190.625], [533.625, 301.875], [452.375, 460.625],
                     [558.375, 188.625], [342.444, 362.596], [345.625, 41.875], [341.625, 146.125]])
    dst2 = np.array([[204.780, 92.367], [201.875, 190.625], [297.125, 296.875], [224.446, 456.556],
                     [318.407, 192.155], [107.625, 371.375], [109.875, 26.624], [106.625, 138.125]])
    homo2 = np.array([[1.738134e+00, 9.499230e-03, -4.466950e+02],
                      [2.745438e-01, 1.514702e+00, -1.213722e+02],
                      [1.160017e-03, 2.073298e-05, 1.000000e+00]])

    correct1 = np.array([0., 1.25, 2.5])
    correct2 = np.array([0.89333816, 2.72870971, 0.54554879,
                         0.14323777, 0.12301613, 0.57911088, 0.3172226, 0.69924904])
    error2 = compute_homography_error(src2, dst2, homo2)

    assert np.sum(error1 - correct1) < 1e-8, STR_OUTPUT_MISMATCH+":case 1"
    assert np.sum(error2 - correct2) < 1e-8, STR_OUTPUT_MISMATCH+":case 2"


def compute_homography_ransac_test(data_dir):
    """
        Implement the following function(s): compute_homography_ransac()
    """
    #
    im1 = load_image(os.path.join(data_dir, 'pano_2.jpg'))
    im2 = load_image(os.path.join(data_dir, 'pano_3.jpg'))

    im1_points = np.array([[434.375, 93.625], [429.625, 190.625],
                           [533.625, 301.875], [452.375, 460.625],
                           [558.375, 188.625], [342.444, 362.596],
                           [345.625, 41.875], [341.625, 146.125],
                           [424.375, 385.375], [602.875, 183.125]])  # last row is outliers

    im2_points = np.array([[204.780, 92.367], [201.875, 190.625],
                           [297.125, 296.875], [224.446, 456.556],
                           [318.407, 192.155], [107.625, 371.375],
                           [109.875, 26.624], [106.625, 138.125],
                           [514.526, 142.354], [348.375, 304.375]])  # last row is outliers

    vis = draw_matches(im1, im2, im1_points, im2_points)

    plt.figure(figsize=(16, 8))
    plt.imshow(vis)

    # Example1: Wrap without outliers
    # exclude the last 2 wrong correspondences
    im1_points_inliers = im1_points[:-2, :]
    im2_points_inliers = im2_points[:-2, :]
    homo_clean = compute_homography(im1_points_inliers, im2_points_inliers)
    stitched_clean = warp_images_all([im1, im2], [homo_clean, np.eye(3)])
    plt.figure(figsize=(12, 8))
    plt.imshow(stitched_clean)
    plt.title("Without outliers")
    plt.show()
    print('Computed clean homography matrix:\n', homo_clean)

    # Example2: Wrap with outliers
    homo_outlier = compute_homography(im1_points, im2_points)
    stitched_outlier = warp_images_all([im2, im1], [np.eye(3), homo_outlier])
    plt.figure(figsize=(12, 8))
    plt.imshow(stitched_outlier)
    plt.title("With outliers")
    plt.show()

    # Implement the robust homography computation which can handle outliers, i.e. RANSAC
    homo_estimated, mask = compute_homography_ransac(im1_points, im2_points)
    # Visualization of stitched images
    stitched_estimated = warp_images_all(
        [im1, im2], [homo_estimated, np.eye(3)])
    plt.figure(figsize=(12, 8))
    plt.title("Roust estimation with RANSAC")
    plt.imshow(stitched_estimated)

    vis = draw_matches(im1, im2, im1_points, im2_points, inlier_mask=mask)
    plt.figure(figsize=(16, 8))
    plt.imshow(vis)
    plt.show()

    print('Computed homography matrix with RANSAC:\n', homo_estimated)


def main():

    dataset = os.path.join(DATA_DIR, DATASET)
    os.makedirs(PREDICTION_DIR, exist_ok=True)

    # Part 1: Robust Homography Estimitation

    # Robust Homography Estimation using RANSAC
    # Implement the following function(s): compute_homography_error()
    # step 1: compute_homography_error()
    compute_homography_error_test()
    # step2: Implement the following function(s): compute_homography(),compute_homography_ransac()
    compute_homography_ransac_test(dataset)


if __name__ == '__main__':
    main()
