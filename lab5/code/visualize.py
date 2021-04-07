import os
import numpy as np
from preprocess import RESULT_DIR
import open3d as o3d


def visualize_point_cloud(pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw([pcd])


def main():
    points3d_save_file = os.path.join(RESULT_DIR, 'points3d.npy')
    points3da = np.load(
        "C:\\Users\\Hans Kurnia\\Desktop\\HANS\\AY2021 Sem 2\\CS4277\\CS4277_Labs\\lab5\\code\\predictions\\mini-temple\\results\\no-bundle-adjustment\\points3d.npy")
    visualize_point_cloud(pts=points3da)
    points3db = np.load(
        "C:\\Users\\Hans Kurnia\\Desktop\\HANS\\AY2021 Sem 2\\CS4277\\CS4277_Labs\\lab5\\code\\ta-results\\mini-temple\\results\\no-bundle-adjustment\\points3d.npy")
    visualize_point_cloud(pts=points3db)
    print(
        f"Are the mini-temple 3d points equal?: {np.array_equal(points3da, points3db)}")
    points3d = np.load(
        "C:\\Users\\Hans Kurnia\\Desktop\\HANS\\AY2021 Sem 2\\CS4277\\CS4277_Labs\\lab5\\code\\predictions\\mini-temple\\results\\bundle-adjustment\\points3d.npy")
    visualize_point_cloud(pts=points3d)

    points3d = np.load(
        "C:\\Users\\Hans Kurnia\\Desktop\\HANS\\AY2021 Sem 2\\CS4277\\CS4277_Labs\\lab5\\code\\ta-results\\mini-temple\\results\\bundle-adjustment\\points3d.npy")
    visualize_point_cloud(pts=points3d)
    points3da = np.load(
        "C:\\Users\\Hans Kurnia\\Desktop\\HANS\\AY2021 Sem 2\\CS4277\\CS4277_Labs\\lab5\\code\\predictions\\temple\\results\\no-bundle-adjustment\\points3d.npy")
    visualize_point_cloud(pts=points3da)
    points3db = np.load(
        "C:\\Users\\Hans Kurnia\\Desktop\\HANS\\AY2021 Sem 2\\CS4277\\CS4277_Labs\\lab5\\code\\ta-results\\temple\\results\\no-bundle-adjustment\\points3d.npy")
    visualize_point_cloud(pts=points3db)
    print(
        f"Are the temple 3d points equal?: {np.array_equal(points3da, points3db)}")


if __name__ == '__main__':
    main()
