"""HW09: detect and track one moving animal using optical flow.

Submit only this file. Use the course cv Conda environment.
Allowed flow methods: cv2.calcOpticalFlowPyrLK,
cv2.calcOpticalFlowFarneback, or DISOpticalFlow_create(...).calc(...).
You may add helpers here. Keep experiments and file I/O out of the function.
"""
import cv2
import numpy as np

def preprocess(frame):
    print("Preprocess complete")
    # convert to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grey

def find_feature_points(frame, box):
    print("Finding Feature Points")
    # use good features to track
    #corners = cv2.goodFeaturesToTrack(frame, box)
    # instead create a mask to apply in good features to track
    x, y, w, h = box
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1) 
    corners = cv2.goodFeaturesToTrack(frame, maxCorners=300, qualityLevel=0.01, minDistance=10, mask = mask)
    # Keeps points where the smaller eigenvalue of 𝐴𝑇𝐴 is large: the windows where Lucas–Kanade is solvable
    return corners

# instead of locating image from frame, we can use previos and current frame??
def find_image(prev, curr):
    print("Locating intiial image")
    # find initial bounding box for the animal
    # THE GOAL OF THIS FUNCTION IS TO RETURN AN INTIIAL BOX
    h, w = prev.shape[:2]
    # pass through the helper to get corners
    corners = find_feature_points(prev, (0, 0, w, h))
    good_prev, good_next = track_points(prev, curr, corners)
    if (good_next is None or len(good_next) < 3):
        return None

    # reshape for index error fix? will it work
    good_next = good_next.reshape(-1, 2)

    x_min = int(good_next[:, 0].min())
    y_min = int(good_next[:, 1].min())
    x_max = int(good_next[:, 0].max())
    y_max = int(good_next[:, 1].max())

    h_updated = y_max - y_min
    w_updated = x_max - x_min

    return (x_min, y_min, w_updated, h_updated)

def track_points(prev, curr, corners):
    # Return [np.nan, np.nan, np.nan, np.nan] when no animal has been detected

    # Pass the previous and current grayscale frames to cv2.calcOpticalFlowPyrLK()
    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev, curr, corners, None, winSize=(15,15))
    prev_points, prev_status, _ = cv2.calcOpticalFlowPyrLK(curr, prev, next_points, None, winSize=(15,15))
    returned = np.linalg.norm(corners - prev_points, axis=2).ravel() < 1
    tracked = (status.ravel() == 1) & (prev_status.ravel() == 1) & returned
    good_prev_points = corners[tracked]
    good_next_points = next_points[tracked]
    return good_prev_points, good_next_points

# i dont think i need this anymore cuz its in track points
# def find_bad_points(prev_points, next_points, status):
#     pass

# this function should be good
def find_bounding_box(prev_box, prev_points, next_points):
    print("Estimating new boundary box")
    if len(prev_points) == 0:
        return prev_box
    # estimated the new boundary box
    x, y, w, h = prev_box
    shift = next_points - prev_points
    x_shift, y_shift = np.median(shift, axis=0).ravel()
    new_x = x + x_shift
    new_y = y + y_shift
    return (new_x, new_y, w, h)

def track_animal(frames):
    """Return one [x, y, width, height] box for each input frame.

    frames: tuple of BGR uint8 arrays, each shaped (685, 1280, 3),
            in chronological order at 10 frames per second.
            Do not modify the supplied arrays; copy them if needed.
    returns: real numeric NumPy array of shape (len(frames), 4).
             Coordinates refer to the supplied image, not a resized image.
             Use [nan, nan, nan, nan] for a missing detection.

    No initial box is supplied. Use one algorithm and parameter policy for
    every clip. You may keep an estimate when the animal briefly stops.

    Input: a tuple of BGR uint8 NumPy images in chronological order, each with shape (685, 1280, 3), sampled at 10 FPS. 
    The arrays are read-only. Make a copy if you need to modify an image.

    Output: a real numeric NumPy array with shape (len(frames), 4). Row i contains the bounding box for input frame i: [x, y, width, height]

    Return [np.nan, np.nan, np.nan, np.nan] when no animal has been detected.

    Choose one of the following:

Method	Required OpenCV API
Lucas–Kanade	cv2.calcOpticalFlowPyrLK(...)
Farnebäck	cv2.calcOpticalFlowFarneback(...)
Dense Inverse Search (DIS)	cv2.DISOpticalFlow_create(...), followed by .calc(...)
cv2.DISOpticalFlow.create(...) is also an accepted spelling of the DIS factory. Any DIS preset or parameter settings are permitted. Other optical-flow estimators are not allowed.

The chosen optical-flow output must actually contribute to detecting and tracking the animal.
Calling an allowed function and ignoring its result does not satisfy the requirement. 
The evaluator checks for a successful call to an allowed optical-flow estimator; source code may also be 
reviewed to verify meaningful use and compliance. Dummy calls do not earn credit for a prohibited solution.

        Hints
        Visualize motion estimates and boxes throughout the clip, including turns and pauses.
        Check how your approach behaves as the animal’s apparent size changes.
        Experiment with parameters using the same policy throughout the clip.
        Optical flow describes moving pixels or features; deciding which belong to the animal is part of your implementation.

        [x, y, width, height] = [181, 337, 270, 225]. Return a box for each frame; the figure uses a public reference annotation.

        Notes:
        A point at (𝑥,𝑦) moves to (𝑥+𝑢,𝑦+𝑣), giving a displacement vector (𝑢,𝑣)
        Solving for (𝑢,𝑣), Least squares: 𝐴𝑇𝐴𝐝=𝐴𝑇𝐛⇒𝐝=(𝐴𝑇𝐴)−1𝐴𝑇𝐛

        Lucas–Kanade (one window)

                Compute 𝐼𝑥,𝐼𝑦 from a smoothed frame and 𝐼𝑡 from the frame difference.
                Sum products over the window to form 𝐴𝑇𝐴
                and 𝐴𝑇𝐛
                .
                Skip the window if the smaller eigenvalue of 𝐴𝑇𝐴       is too small.
                Solve the 2×2
                system for (𝑢,𝑣)
                .

    """
    # output array
    boxes = np.full((len(frames), 4), np.nan, dtype=float)
    # TODO: detect the moving animal and track it using one allowed flow method.
    # we can use the code on slide 25 for reference but instead we are tracking points of the animal
    # make fake box for now 
    #box = [181, 337, 270, 225]
    box = None
    boxes[0] = box
    prev_frame = preprocess(frames[0])
    print("Loaded prev frame, starting next frame processing")
    for i in range(1, len(frames)):
        # Track your feature points
        # Update your bounding box
        # Store the updated box in boxes[i]
        # preprocess
        #prev_frame = preprocess(frames[i - 1])
        curr_frame = preprocess(frames[i])
        # check id nothing has been found which should be initially, then use find image to establish intial bounding box
        # then if box is not none, provide all veriables to helper functions find feat points, track pts, and find bounding box
        if box is None:
            box = find_image(prev_frame, curr_frame)
        else:
            corners = find_feature_points(prev_frame, box)
            good_prev, good_next = track_points(prev_frame, curr_frame, corners)
            box = find_bounding_box(box, good_prev, good_next)
        # call all of our helper functions
        # corners = find_feature_points(prev_frame, box)
        # good_prev, good_next = track_points(prev_frame, curr_frame, corners)
        # new_box = find_bounding_box(box, good_prev, good_next)
        if box is not None:
            # keep the box inside the 1280x685 image - hw 
            x, y, w, h = box
            w = min(max(w, 10), 1280)
            h = min(max(h, 10), 685)
            x = min(max(x, 0), 1280 - w)
            y = min(max(y, 0), 685 - h)
            box = (x, y, w, h)
            boxes[i] = box
        prev_frame = curr_frame
    return boxes
