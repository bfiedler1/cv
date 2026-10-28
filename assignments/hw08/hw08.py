"""HW08: Document corners. Due Wednesday, October 28, 2026.

Complete detect_document. Keep intersect_lines unchanged.
Submit only hw08.py. Run: python evaluate.py
"""
import cv2
import numpy as np


def intersect_lines(line1, line2):
    """Intersect two Hough (r, theta) lines; return (x,y), or None if parallel."""
    r1, t1 = line1
    r2, t2 = line2
    matrix = np.array([[np.cos(t1), np.sin(t1)],
                       [np.cos(t2), np.sin(t2)]], dtype=float)
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    return np.linalg.solve(matrix, np.array([r1, r2], dtype=float))

def sample_brightness(image, line, center, offset):
    """
    * Googled pipeline to get this done * 
    Determine Direction: Find the vector from the image center to the line. 
    "Toward the center" is your inside region (should be lighter document page), 
    and "away from the center" is your outside region (should be a darker background).
    
    Sample Points: Pick N equidistant points along the detected line segment.
    Check Gradients: For each point, sample pixel values slightly to the left/right (or above/below) perpendicular to the line.
    Score the Line: Favor lines where the inner sample is consistently brighter than the outer sample. 
    You can score this by counting how many sample locations satisfy (Intensity_{in} > Intensity_{out}
    or by averaging the local gradients.
    """
    h, w = image.shape
    r, theta = line
    cos = np.cos(theta)
    sin = np.sin(theta)
    # get the center of the image
    # it can point toward OR away from the center.
    # center_x*cos + center_y*sin is the center's own r value (r = x cos(t) + y sin(t))
    center_x, center_y = center
    sign = 1 if (center_x * cos + center_y * sin - r) > 0 else -1
    total_score = 0
    valid = 0
    # try out values to move along line
    for t in range(-1500, 1500, 4):
        # get point of the line
        # t is the distance along the line
        x = r * cos - t * sin 
        y = r * sin + t * cos
        # direction and coordinates of the vector - checking gradients
        xi, yi = int(x + sign * offset * cos), int(y + sign * offset * sin) 
        xo, yo = int(x - sign * offset * cos), int(y - sign * offset * sin) 
        # making sure that all sample points are within boundaries
        if 0 <= xi < w and 0 <= yi < h and 0 <= xo < w and 0 <= yo < h:
            # score the line by taking outside - inside 
            total_score += float(image[yi, xi]) - float(image[yo, xo])
            valid += 1
    # this last part i used this from googling 
    # score = total_score / valid_samples if valid_samples > 0 else -np.inf
    # scores.append(score)
    if not valid > 0: 
        return -np.inf
    return total_score / valid

def best_line(image, candidates, center):
    """calculate the best score out of the candidates outside - inside
    no line and no score to start 
    """
    best = None
    best_score = -np.inf
    center_x, center_y = center
    # for every line in candidates use the brightness ehlper to score 
    for r, theta in candidates:
        score = sample_brightness(image, (r, theta), (center_x, center_y), 5)
        # best line based on prev score
        if score > best_score:
            best, best_score = (r, theta), score
    return best

def detect_document(image):
    """Return a numeric NumPy array of four (x, y) corners, shape (4, 2).

    Input: BGR uint8 image, shape (1000, 750, 3). Do not modify it.
    Corner order is arbitrary; use original input coordinates.
    Use cv2.HoughLines and the provided intersect_lines helper.
    Return np.empty((0, 2)) if no document can be detected.
    Additional helper functions are allowed in this file.
    """
    '''
    Your pipeline must:
        Preprocess the image and compute an edge image.
        Call cv2.HoughLines to generate candidate boundary lines.
        Select four distinct document boundaries from the candidates.
        Call the provided intersect_lines helper to compute the four returned corners from appropriate pairs of selected lines.

    If detection fails, return np.empty((0, 2))
    The cv2.HoughLines function returns a 3D NumPy array of shape (N, 1, 2)
    '''
    # 1) PREPROCESS THE IMAGE AND COMPUTE AN EDGE IMAGE
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #h, w = img_grey.shape
    img_smooth = cv2.GaussianBlur(img_grey, (5, 5), 0)
    img_edges = cv2.Canny(img_smooth, 40, 100)
    height, width = img_grey.shape
    center_x = width / 2
    center_y = height / 2

    # 2) finds the candidates for boundary lines
    # raise threshold to reject weak candidates
    # returns an array of lines
    lines = cv2.HoughLines(img_edges, rho=1, theta=np.pi / 180, threshold=120)
    if lines is None:
        return np.empty((0, 2))

    # 3) SELECT FOUR DISTINCT BOUNDARIES FOR THE DOCUMENT
    # therefore, we need 4 sides, 2 for each orinetation - h and v
    # then we cna parse through lines that way
    # both of these needs to be 2 at the end
    #horizontal_lines = []
    #vertical_lines = []
    # try a different approach
    left_lines = []
    right_lines = []
    top_lines = []
    bottom_lines = []
    min_dist_vertical = 180
    min_dist_horizontal = 250

    # take out the lines
    # leftside = vertical_lines[0]
    # rightside = vertical_lines[-1]
    # topline = horizontal_lines[0]
    # bottomline = horizontal_lines[-1]
    # take out the lines
    # leftside = left_lines[0]
    # rightside = right_lines[-1]
    # topline = top_lines[0]
    # bottomline = bottom_lines[-1]

    for line in lines:
        r, theta = line[0]
        cos = np.cos(theta)
        sin = np.sin(theta)
        # how far this line is from the image center (perpendicular distance)
        dist = abs(r - (center_x * cos + center_y * sin))

        if abs(theta - np.pi / 2) < np.pi / 4:
            # horizontal line where it crosses center
            if dist < min_dist_horizontal:
                continue
            y_center = (r - center_x * cos) / sin
            if y_center < center_y:
                top_lines.append((r, theta))
            else:
                bottom_lines.append((r, theta))
        else:
            # vertical line where it crosses center
            if dist < min_dist_vertical:
                continue
            x_center = (r - center_y * sin) / cos
            if x_center < center_x:
                left_lines.append((r, theta))
            else:
                right_lines.append((r, theta))

    if not (left_lines and right_lines and top_lines and bottom_lines):
        return np.empty((0, 2))

    # pick the best line on each side using helper score funciton
    leftside = best_line(img_smooth, left_lines, (center_x, center_y))
    rightside = best_line(img_smooth, right_lines, (center_x, center_y))
    topline = best_line(img_smooth, top_lines, (center_x, center_y))
    bottomline = best_line(img_smooth, bottom_lines, (center_x, center_y))

    if None in (leftside, rightside, topline, bottomline):
        return np.empty((0, 2))

    # calculate the corners between the selected borders to the document
    corners = [
        intersect_lines(leftside, topline),
        intersect_lines(rightside, topline),
        intersect_lines(rightside, bottomline),
        intersect_lines(leftside, bottomline),
    ]

    if any(p is None for p in corners):
        return np.empty((0, 2))

    corners = np.array(corners, dtype=float).reshape(4, 2)
    
    return corners

    #for line in lines:
    #    r, theta = line[0]

    # # corner = intersect_lines()

    # # The first four returned lines may not be the document boundaries. To rank candidates, sample brightness on both sides at several locations along each line. Favor a consistent change from lighter pixels toward the image center to darker pixels away from it, since the document contains the center.
    # # Group candidates whose visible segments nearly overlap, keeping one representative per group. Then select four distinct boundaries surrounding the image center. Similar orientation alone does not make two lines duplicates.
    # # call a brightness helper to score
    # best_brightness = []
    # for line in lines:
    #     for j in range(i + 1, len(lines)):
    #         corner = intersect_lines(lines[i][0], lines[j][0])
    #         if corner is not None:
    #             corners.append(corner)
    #         else:
    #             return np.empty((0, 2))