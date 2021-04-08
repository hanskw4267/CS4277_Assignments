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
    points3d = np.load(points3d_save_file)
    visualize_point_cloud(pts=points3d)


if __name__ == '__main__':
    main()
