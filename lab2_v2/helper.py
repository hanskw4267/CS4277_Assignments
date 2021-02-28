from math import floor, ceil, sqrt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv


_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)


"""Helper functions: You should not have to touch the following functions.
"""
def load_image(im_path):
    """Loads image and converts to RGB format

    Args:
        im_path (str): Path to image

    Returns:
        im (np.ndarray): Loaded image (H, W, 3), of type np.uint8.
    """
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def draw_matches(im1, im2, im1_pts, im2_pts, inlier_mask=None):
    """Generates a image line correspondences

    Args:
        im1 (np.ndarray): Image 1
        im2 (np.ndarray): Image 2
        im1_pts (np.ndarray): Nx2 array containing points in image 1
        im2_pts (np.ndarray): Nx2 array containing corresponding points in
          image 2
        inlier_mask (np.ndarray): If provided, inlier correspondences marked
          with True will be drawn in green, others will be in red.

    Returns:

    """
    height1, width1 = im1.shape[:2]
    height2, width2 = im2.shape[:2]
    canvas_height = max(height1, height2)
    canvas_width = width1 + width2

    canvas = np.zeros((canvas_height, canvas_width, 3), im1.dtype)
    canvas[:height1, :width1, :] = im1
    canvas[:height2, width1:width1+width2, :] = im2

    im2_pts_adj = im2_pts.copy()
    im2_pts_adj[:, 0] += width1

    if inlier_mask is None:
        inlier_mask = np.ones(im1_pts.shape[0], dtype=np.bool)

    # Converts all to integer for plotting
    im1_pts = im1_pts.astype(np.int32)
    im2_pts_adj = im2_pts_adj.astype(np.int32)

    # Draw points
    all_pts = np.concatenate([im1_pts, im2_pts_adj], axis=0)
    for pt in all_pts:
        cv2.circle(canvas, (pt[0], pt[1]), 4, _COLOR_BLUE, 2)

    # Draw lines
    for i in range(im1_pts.shape[0]):
        pt1 = tuple(im1_pts[i, :])
        pt2 = tuple(im2_pts_adj[i, :])
        color = _COLOR_GREEN if inlier_mask[i] else _COLOR_RED
        cv2.line(canvas, pt1, pt2, color, 2)

    return canvas

def Line_Equation(point1,point2):
    '''
    # Ax+By+C=0
    Args:
        point1: The first point coordinate (x1,y1)
        point2: The second point coordinate (x2,y2)
    Returns: Homogeneous 3-vectors (A,B,C)
    '''
    # x1, y1=point1.x,point1.y
    # x2, y2=point2.x,point2.y
    # norm = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
    # A=(y2-y1)
    # B=(x1-x2)
    # C=(x2*y1-x1*y2)
    line=np.cross(np.array([point1.x,point1.y,1]),np.array([point2.x,point2.y,1]))

    line=line[:]/line[-1]
    return line#(A,B,C)


class Point:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.coordinate=(x,y)
    def __sub__(self,point):
        return Point(self.x-point.x,self.y-point.y)

class Line:
    def __init__(self,point1,point2):
        self.point1=point1
        self.point2=point2
        self.vec_para=Line_Equation(point1,point2)
    def cross_product(self):
        #[0,0,point1.x * point2.y - point2.x * point1.y]
        vector1=[self.point1.x,self.point1.y,1]
        vector2=[self.point2.x,self.point2.y,1]
        return np.cross(vector1,vector2)

    def cross_line(self,line2):
        diff_vec1=line2.point1-self.point1
        diff_vec2=line2.point2-self.point1
        diff_vec=self.point2-self.point1

        if self.cross_product(diff_vec1,diff_vec)[-1]*self.cross_product(diff_vec2,diff_vec[-1])<=0:
            return True
        else:
            return False
    def is_cross(self,line2):
        if self.cross_line(line2) and line2.cross_line(self):
            return True
        else:
            return False
    def intersetion_point(self,line2):
        '''
        Given two lines (parameterized as homogeneous 3-vectors (Ax+By+C=0)), return the intersection points (x,y)
        Args:
            line2:the second line (A2,B2,C2)
        Returns: the intersection point (X,Y)
        '''

        X, Y = None, None
        A1, B1, C1 = self.vec_para
        A2, B2, C2 = line2.vec_para
        D = A1 * B2 - A2 * B1
        if D == 0:
            print("No intersection points!")
        else:
            X = (B1 * C2 - B2 * C1) / D
            Y = (A2 * C1 - A1 * C2) / D
        inter = np.cross(self.vec_para, line2.vec_para)
        inter = inter / inter[2]
        return Point(int(X), int(Y))

