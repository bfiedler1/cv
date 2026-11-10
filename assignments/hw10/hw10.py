"""HW10: Camera simulation and calibration. Due Wednesday, November 11, 2026.

Set SQUARE_SIZE_MM and complete the four functions below.
Submit hw10.py and your photos.
Run:  python evaluate.py projection
      python evaluate.py calibration photos
"""
import cv2
import numpy as np

# TODO: side length of one square of YOUR printed or displayed checkerboard, in mm.
SQUARE_SIZE_MM = 20


def project_points(X_w, K, R_c_w, t_c_w):
    """Return the pixel coordinates (u, v) of N world points, shape (N, 2).

    X_w:   (N, 3) array; row i is the world point ^wX_i.
    K:     (3, 3) intrinsic matrix.
    R_c_w: (3, 3) rotation ^cR_w from the world frame to the camera frame.
    t_c_w: (3, 1) translation ^ct_w from the world frame to the camera frame.

    Use NumPy. Do not call cv2.projectPoints in this function.
    """
    # TODO: world frame -> camera frame -> pixels (divide by the depth Z_c).
    # world frame -> camera frame
    # 𝑐𝐗=𝑐𝐑𝑤𝑤𝐗+𝑐𝐭𝑤
    X_c = R_c_w @ X_w.T + t_c_w
    print(X_w.shape, X_c.shape)
    # 𝑐𝐗=(𝑋𝑐,𝑌𝑐,𝑍𝑐)
    # camera -> pixels
    X = K @ X_c
    # 𝑥=𝑓𝑋𝑐/𝑍𝑐,  𝑦=𝑓𝑌𝑐/𝑍𝑐
    x = X[0]/X[2]
    y = X[1]/X[2]
    return np.array([x, y]).T


def board_points(pattern_size, square_size):
    """Return the board points ^bX_i = (X, Y, 0), float32, shape (columns*rows, 3).

    pattern_size = (columns, rows) counts INNER corners, e.g., (8, 6).
    Start at (0, 0, 0) and go row by row: X increases fastest, then Y.
    Neighboring corners are square_size apart.
    """
    # TODO
    cols, rows = pattern_size
    # Start at (0, 0, 0)
    # np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # Neighboring corners are square_size apart
    return pts * square_size


def detect_board(image, pattern_size):
    """Return the refined corners from cv2.cornerSubPix, or None if not found.

    image: BGR uint8 photo. Do not modify it.
    Use cv2.findChessboardCorners, then refine with cv2.cornerSubPix.
    Return the refined corners as OpenCV returns them: float32, one (x, y)
    per corner, columns*rows corners in OpenCV's order.
    """
    # TODO
    '''
    detect_board(image, pattern_size) finds the inner corners 𝐱𝑖 in a BGR uint8 photo:

Convert the photo to grayscale. Do not modify the input image.
Call cv2.findChessboardCorners. If the board is not found, return None.
You must refine the corners with cv2.cornerSubPix and return the refined corners as OpenCV returns them: 
a float32 array with one (𝑥,𝑦) pixel coordinate for each of the columns*rows corners, in OpenCV’s order. 
If you resize the photo for detection, convert the corners back to the input photo’s coordinates.
    '''
    # Convert the photo to grayscale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Call cv2.findChessboardCorners. If the board is not found, return None.
    found, board_corners = cv2.findChessboardCorners(grey, pattern_size)
    if board_corners is None or not found:
        return None
    # You must refine the corners with cv2.cornerSubPix and return the refined corners as OpenCV returns them:
    refined_corners = cv2.cornerSubPix(grey, board_corners.copy(), (11,11), (-1,1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return refined_corners


def calibrate_camera(images, pattern_size, square_size):
    """Calibrate the camera from checkerboard photos.

    images: a list of BGR uint8 photos, all the same size.
            Loop over it with `for image in images:`.
    Skip photos in which detect_board returns None.

    Return the five outputs of cv2.calibrateCamera, unchanged:
        rms, K, dist, rvecs, tvecs
    """
    '''
    calibrate_camera(images, pattern_size, square_size) calibrates the camera from all photos:

Get the board points with your board_points.
Loop over the photos with for image in images:. Call your detect_board on each photo and skip it if the result is None. Otherwise, append the board points to obj_points and the corners to img_points.
Call cv2.calibrateCamera(obj_points, img_points, image_size, None, None), where image_size is (width, height) of the photos. Note the order: NumPy’s image.shape is (height, width, 3).
Return the five outputs of cv2.calibrateCamera unchanged: rms, K, dist, rvecs, tvecs.
    '''
    # TODO: use board_points, detect_board, and cv2.calibrateCamera.
    points = board_points(pattern_size, square_size)
    # append the board points to obj_points and the corners to img_points.
    obj_points = []
    img_points = [] 
    # Loop over the photos with for image in images:
    for image in images:
        corners = detect_board(image, pattern_size)
        # Call your detect_board on each photo and skip it if the result is None
        if corners is None:
            continue
        # append the board points to obj_points and the corners to img_points.
        img_points.append(corners)
        obj_points.append(points)
    image_size = (images[0].shape[1], images[0].shape[0])
    # Call cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    return rms, K, dist, rvecs, tvecs
