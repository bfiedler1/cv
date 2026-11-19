"""HW11: 3D reconstruction from two views. Due Monday, November 23, 2026.

Complete the three functions below. Submit hw11.py, your two photos, and calibration.json.
Run:  python evaluate.py
      python evaluate.py --photos photos
"""
import cv2
import numpy as np


def reconstruct_sparse(image1, image2, K, dist):
    """Relative pose and sparse 3D points from two photos.

    image1, image2: BGR uint8 photos of the same size.
    K: (3, 3) intrinsic matrix.  dist: lens distortion coefficients.

    Return pts1, pts2, R, t, X:
        pts1, pts2: (N, 2) matched pixels in image1 and image2 (as given, distorted)
        R: (3, 3) and t: (3, 1) pose of camera 2 relative to camera 1, X_2 = R X_1 + t
        X: (N, 3) triangulated points in camera 1's frame; row i belongs to match i
    """
    '''
    Detect and match features between the two photos (e.g., SIFT with the ratio test).
    Remove lens distortion from the matched points with cv2.undistortPoints(..., P=K).
    Estimate the essential matrix with cv2.findEssentialMat using RANSAC.
    Recover the relative pose with cv2.recoverPose, and keep its inliers.
    Triangulate the inliers with cv2.triangulatePoints, with camera 1 as the reference frame.
    '''
    # TODO: match features, undistort the matches, cv2.findEssentialMat (RANSAC),
    #       cv2.recoverPose, cv2.triangulatePoints.
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    good = [m for m, n in cv2.BFMatcher().knnMatch(des1, des2, k=2) if m.distance < 0.75 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    pts1_rect = cv2.undistortPoints(pts1, K, dist, P=K).reshape(-1, 2)
    pts2_rect = cv2.undistortPoints(pts2, K, dist, P=K).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(pts1_rect, pts2_rect, K, method=cv2.RANSAC)
    _, R, t, mask = cv2.recoverPose(E, pts1_rect, pts2_rect, mask)
    
    X = cv2.triangulatePoints(pts1, pts2, pts1_rect, pts2_rect)
    return pts1, pts2, R, t, X


def rectify_pair(image1, image2, K, dist, R, t):
    """Rectify both photos using the pose R, t of camera 2 relative to camera 1.

    Return rect1, rect2, R1, R2, P1, P2, Q:
        rect1, rect2: rectified BGR uint8 images, the same size as the inputs
        R1, R2, P1, P2, Q: the outputs of cv2.stereoRectify
    """
    # TODO: cv2.stereoRectify, cv2.initUndistortRectifyMap, cv2.remap. - should be done
    h, w = image1.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, dist, K, dist, (w, h), R, t, alpha = 1)
    map1 = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
    map2 = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_32FC1)
    rect1 = cv2.remap(image1, *map1, cv2.INTER_LINEAR)
    rect2 = cv2.remap(image2, *map2, cv2.INTER_LINEAR)
    return rect1, rect2, R1, R2, P1, P2, Q


def reconstruct_dense(rect1, rect2, Q):
    """Disparity and dense 3D points from a rectified pair.

    Return disparity, valid, points:
        disparity: (H, W) float32 disparity in pixels
        valid: (H, W) Boolean, True where the 3D point is usable (finite, positive depth)
        points: (H, W, 3) 3D points in rectified camera 1's frame
    """
    '''
    Compute the disparity with cv2.StereoBM or cv2.StereoSGBM, and divide it by 16 to get pixels.
    Compute the 3D points with cv2.reprojectImageTo3D and Q.
    '''
    # TODO: cv2.StereoBM or cv2.StereoSGBM, divide by 16, cv2.reprojectImageTo3D.
    grey1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)
    bm = cv2.StereoBM_create(numDisparities=128, blockSize=15)
    disp = bm.compute(grey1, grey2).astype(np.float32) / 16
    disp[disp < 0] = np.nan
    points = cv2.reprojectImageTo3D(disp, Q)
    #colors = left[..., ::-1]  # BGR -> RGB
    valid = ~np.isnan(disp)
    return disp, valid, points
