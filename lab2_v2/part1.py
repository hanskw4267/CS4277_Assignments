import os
from main import *
from helper import *
import tkinter
import tkinter.messagebox
import csv

DATASET = 'inputs'
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PREDICTION_DIR = os.path.join(PROJECT_DIR, 'predictions')
DATASETS = ['']


def _load_dataset_images(dataset: str):
    """ loads the dataset image """
    img_dir = os.path.join(DATA_DIR, DATASET, dataset)
    img = cv2.imread(img_dir)
    return img


def _show_images(image: np.ndarray, win_name: str):
    cv2.namedWindow(win_name, 0)
    cv2.imshow(win_name, image)


def _name_image_window(win_names: list):
    if len(win_names) > 0:
        for win_name in win_names:
            cv2.namedWindow(win_name, 0)
    else:
        os.error("Please input the image window name")


def _save_metric_rectified_image(image: np.ndarray, save_file: os.path):
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    cv2.imwrite(save_file, image)


def _get_points_from_files(save_file: os.path):
    lines_vec = []

    # Read points from the csv file and parameterize line to homogeneous 3-vectors
    with open(save_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        filenames = next(reader)
        csv_reader = csv.DictReader(csvfile, fieldnames=filenames)
        for row in csv_reader:

            point1 = Point(eval(row['point1'])[0], eval(row['point1'])[1])
            point2 = Point(eval(row['point2'])[0], eval(row['point2'])[1])
            cur_line_vec = Line(point1, point2)
            lines_vec.append(cur_line_vec)

    return lines_vec


def _visiual_lines_in_image(src_img: np.ndarray, lines_vec: list):
    for i in range(0, len(lines_vec)):
        cv2.line(src_img, lines_vec[i].point1.coordinate,
                 lines_vec[i].point2.coordinate, (0, 255, 255), 5)


def _mouse_event(event, x, y, flags, param):

    image, save_file, CoordinateX, CoordinateY = param[0], param[1], param[2], param[3]

    if event == cv2.EVENT_LBUTTONDOWN:  # & flags<3:

        xy = "%d,%d" % (x, y)
        cv2.circle(image, (x, y), 2, (0, 0, 255), thickness=-1)
        cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 0, 0), thickness=2)
        CoordinateX.append(x)
        CoordinateY.append(y)

    elif event == cv2.EVENT_RBUTTONDOWN:
        csvFile = open(save_file, "w", encoding='utf8', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(['lines', 'point1', 'point2'])
        for i in range(0, len(CoordinateX), 2):
            writer.writerow(
                [str(int(i / 2)), (CoordinateX[i], CoordinateY[i]), (CoordinateX[i + 1], CoordinateY[i + 1])])
        csvFile.close()


def _confirm_select_constraints(str=''):

    root = tkinter.Tk()
    root.withdraw()
    ans = tkinter.messagebox.askokcancel(
        'Note', 'Do you wish to select '+str+' lines on the shown image?')
    return ans


class _select_points_interface:
    def __init__(self, image: np.ndarray, image_window_name: str, save_file: os.path):
        CoordinateX = []  # Coordinate Set for X
        CoordinateY = []  # Coordinate Set for Y
        show_img = image.copy()
        params = [show_img, save_file, CoordinateX, CoordinateY]
        cv2.namedWindow(image_window_name, 0)
        cv2.setMouseCallback(
            image_window_name, self._mouse_event, param=params)

        while True:
            cv2.imshow(image_window_name, show_img)

            if len(CoordinateX) % 2 == 0 and len(CoordinateX) > 0:
                cv2.line(show_img, (CoordinateX[-2], CoordinateY[-2]),
                         (CoordinateX[-1], CoordinateY[-1]), (0, 255, 0), 1)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                csvFile = open(save_file, "w", encoding='utf8', newline='')  #
                writer = csv.writer(csvFile)  #
                writer.writerow(['lines', 'point1', 'point2'])
                for i in range(0, len(CoordinateX), 2):
                    writer.writerow(
                        [str(int(i / 2)), (CoordinateX[i], CoordinateY[i]), (CoordinateX[i + 1], CoordinateY[i + 1])])
                csvFile.close()
                break
        cv2.destroyAllWindows()

    def _mouse_event(self, event, x, y, flags, param):

        image, save_file, CoordinateX, CoordinateY = param[0], param[1], param[2], param[3]

        if event == cv2.EVENT_LBUTTONDOWN:  # & flags<3:

            xy = "%d,%d" % (x, y)
            cv2.circle(image, (x, y), 2, (0, 0, 255), thickness=-1)
            cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 0, 0), thickness=2)
            CoordinateX.append(x)
            CoordinateY.append(y)
            #


def transform_homography_test():
    '''
        Implement the following functions: transform_homography()
    '''

    src = np.array([[434.375, 93.625], [429.625, 190.625], [533.625, 301.875], [452.375, 460.625],
                    [558.375, 188.625], [342.444, 362.596], [345.625, 41.875], [341.625, 146.125]])
    dst = np.array([[204.780, 92.367], [201.875, 190.625], [297.125, 296.875], [224.446, 456.556],
                    [318.407, 192.155], [107.625, 371.375], [109.875, 26.624], [106.625, 138.125]])
    homo = np.array([[1.738134e+00, 9.499230e-03, -4.466950e+02],
                     [2.745438e-01, 1.514702e+00, -1.213722e+02],
                     [1.160017e-03, 2.073298e-05, 1.000000e+00]])
    computed_dst = transform_homography(src, homo)

    print("Computed_dst:{}\n Original_dst:{}".format(computed_dst, dst))


def wrap_image_test(data_dir):
    """
        Implement the following function(s): warp_image()
    """

    template = load_image(os.path.join(data_dir, 'hzbook_2.jpg'))
    original = load_image(os.path.join(data_dir, 'hzbook_1.jpg'))
    # homo is the homography matrix that maps points in template to original
    homo = np.array([[9.01221661e-01, -1.79252179e-01, 1.82000000e+02],
                     [-1.39482959e-01, 5.49034320e-01, 9.40000000e+01],
                     [-1.06281109e-05, -7.07553504e-04, 1.00000000e+00]])
    overlaid = warp_image(template, original, homo)

    plt.figure()
    plt.imshow(template)
    plt.title('Template')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(overlaid)
    plt.title('Modified image')
    plt.show()


def _affinely_rectified_test(image, image_name):
    '''
        Implement the following functions: compute_affine_rectification()
    '''
    src_image = image.copy()
    # Interactive interface for constraint selection
    ans = _confirm_select_constraints('Two Parallel')
    if ans:
        _select_points_interface(image=src_image, image_window_name='Original_image',
                                 save_file=os.path.join(DATA_DIR, image_name+'_Parallel_lines.csv'))
    # read points (x1,y1) & (x2,y2) from files to form their line vectors (l1,l2,l3)
    para_lines_vec = _get_points_from_files(
        save_file=os.path.join(DATA_DIR, image_name+'_Parallel_lines.csv'))
    _visiual_lines_in_image(src_img=src_image, lines_vec=para_lines_vec)
    # compute affine rectification (Implement )
    affined_image = compute_affine_rectification(
        src_img=src_image, lines_vec=para_lines_vec)
    # visualization
    # affined_image=cv2.resize(affined_image,src_image.shape[:2])
    _show_images(image=affined_image, win_name='Affinely_rectified_image')

    return affined_image


def _metric_rectified_twostep_test(image, image_name):
    '''
        Implement the following functions: compute_metric_rectification_step2()
    '''
    affinely_rectified_image = image.copy()
    # Interactive interface for constraint selection
    ans = _confirm_select_constraints('Two Orthogonal')
    if ans:
        _select_points_interface(image=affinely_rectified_image, image_window_name='Affine_image',
                                 save_file=os.path.join(DATA_DIR, image_name+'_Orthogonal_2_lines.csv'))
    # read points (x1,y1) & (x2,y2) from files to form their line vectors (l1,l2,l3)
    ortho_2_lines_vec = _get_points_from_files(
        save_file=os.path.join(DATA_DIR, image_name+'_Orthogonal_2_lines.csv'))
    _visiual_lines_in_image(
        src_img=affinely_rectified_image, lines_vec=ortho_2_lines_vec)
    # compute metric rectification
    rectified_image1 = compute_metric_rectification_step2(
        src_img=affinely_rectified_image, line_vecs=ortho_2_lines_vec)
    # visualization
    _show_images(image=rectified_image1, win_name='Metric_rectified_image1')
    return rectified_image1


def _metric_rectified_onestep_test(image, image_name):
    '''
        Implement the following functions: compute_metric_rectification_one_step()
    '''
    src_image = image.copy()
    # Interactive interface for constraint selection
    ans = _confirm_select_constraints('Five Orthogonal')
    if ans:
        _select_points_interface(image=src_image, image_window_name='Original_image',
                                 save_file=os.path.join(DATA_DIR, image_name+'_Orthogonal_5_lines.csv'))
    # read points (x1,y1) & (x2,y2) from files to form their line vectors (l1,l2,l3)
    ortho_5_lines_vec = _get_points_from_files(
        save_file=os.path.join(DATA_DIR, image_name+'_Orthogonal_5_lines.csv'))
    _visiual_lines_in_image(src_img=src_image, lines_vec=ortho_5_lines_vec)
    # compute metric rectification
    rectified_image2 = compute_metric_rectification_one_step(
        src_img=src_image, line_vecs=ortho_5_lines_vec)
    # visualization
    _show_images(image=rectified_image2, win_name='Metric_rectified_image2')
    return rectified_image2


def main():
    dataset = os.path.join(DATA_DIR, DATASET)
    prediction_dir = os.path.join(DATA_DIR, 'predictions')
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    image_name = "door01"
    image_ori = _load_dataset_images(dataset=image_name+".jpg")

    # Preliminary: Fun with Homography
    # 1) transformation using provided homography matrix
    # transform_homography_test()
    # # 1)Image Warping using Homography
    # wrap_image_test(dataset)

    # Part1: Metric rectification (Two-step: Stratification)
    # Interactively select points on image to form the required constraints
    '''
       If you expect to select constraints on your own, just directly click on the shown image to select points.
       When you finish selection, press 'q' in the keyboard to exit the interactive interface. The selected constrains 
       will be save to the path 'data/*.csv' by default. 
    '''
    # step 1: Affinely rectified
    affined_image = _affinely_rectified_test(image_ori, image_name=image_name)
    # step 2: Metric rectified
    rectified_image1 = _metric_rectified_twostep_test(
        image=affined_image, image_name=image_name)

    # Metric rectification (One-step)
    rectified_image2 = _metric_rectified_onestep_test(
        image=image_ori, image_name=image_name)

    # Save results. Please show your results in the report.
    _save_metric_rectified_image(image=affined_image,
                                 save_file=os.path.join(prediction_dir, image_name+'_affinely_rectified_image_1.png'))
    _save_metric_rectified_image(image=rectified_image1, save_file=os.path.join(
        prediction_dir, image_name+'_final_rectified_image_1.png'))
    _save_metric_rectified_image(image=rectified_image2,
                                 save_file=os.path.join(prediction_dir, image_name+'_final_rectified_image_2.png'))

    cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
