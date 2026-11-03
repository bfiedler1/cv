# i am restarting trying to follow this example on the slides

import cv2
import numpy as np

'''

GREEN, RED = (0, 255, 0), (0, 0, 255)  # OpenCV colors are BGR

def track_points(gray1, gray2, points):
    # Lucas-Kanade from gray1 to gray2, then back to gray1.
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, points, None, winSize=(15, 15), maxLevel=3)
    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray2, gray1, next_points, None, winSize=(15, 15), maxLevel=3)
    # Keep points found in both directions that came back within 1 px.
    returned = np.linalg.norm(points - back_points, axis=1) < 1
    tracked = (status.ravel() == 1) & (back_status.ravel() == 1) & returned
    return next_points, tracked

video = cv2.VideoCapture("vtest.avi")
tracks = []  # one track per point: its list of (x, y) positions
gray1 = None
frame_count = 0

while True:
    ok, frame = video.read()
    if not ok:  # end of the video
        break
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Move each track's latest point to gray2; keep its last 10 positions.
    if gray1 is not None and tracks:
        points = np.float32([track[-1] for track in tracks])
        next_points, tracked = track_points(gray1, gray2, points)
        tracks = [(track + [point])[-10:] for track, point, keep
                  in zip(tracks, next_points, tracked) if keep]

    # Every 10 frames, add new points at least 10 px from those already tracked.
    if frame_count % 10 == 0:
        mask = np.full(gray2.shape, 255, np.uint8)
        for track in tracks:
            cv2.circle(mask, np.int32(track[-1]), 10, 0, -1)
        corners = cv2.goodFeaturesToTrack(gray2, maxCorners=300, qualityLevel=0.01,
                                          minDistance=10, blockSize=7, mask=mask)
        tracks += [[point] for point in corners[:, 0]]

    # Draw tails (green) and current positions (red).
    for track in tracks:
        cv2.polylines(frame, [np.int32(track)], False, GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, np.int32(track[-1]), 2, RED, -1, cv2.LINE_AA)
    cv2.imshow("Lucas-Kanade tracking", frame)
    if cv2.waitKey(100) == 27:  # 100 ms per frame (10 fps); Esc quits
        break

    gray1 = gray2
    frame_count += 1

video.release()
cv2.destroyAllWindows()


'''

def preprocess(frame):
    # convert to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grey

# stays the same from the notes in the slides
def track_points(gray1, gray2, points):
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, points, None, winSize=(15, 15), maxLevel=3)
    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray2, gray1, next_points, None, winSize=(15, 15), maxLevel=3)
    returned = np.linalg.norm(points - back_points, axis=1) < 1
    tracked = (status.ravel() == 1) & (back_status.ravel() == 1) & returned
    return next_points, tracked

def track_animal(frames):
    boxes = np.full((len(frames), 4), np.nan, dtype=float)
    # grey1 is prev
    grey1 = None
    prev_box = None
    tracks = []

    # loop through frames
    for i in range(1, len(frames)):
        '''
        ok, frame = video.read()
            if not ok:  # end of the video
                break
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # Move each track's latest point to gray2; keep its last 10 positions.
            if gray1 is not None and tracks:
                points = np.float32([track[-1] for track in tracks])
                next_points, tracked = track_points(gray1, gray2, points)
                tracks = [(track + [point])[-10:] for track, point, keep
                          in zip(tracks, next_points, tracked) if keep]
        '''
        grey2 = preprocess(frames[i])

        if (grey1 is not None and tracks):
            points = np.float32([t[-1] for t in tracks])
            next_points, tracked = track_points(grey1, grey2, points)
            tracks = [(t + [p])[-10:] for t, p, keep
                      in zip(tracks, next_points, tracked) if keep]

        '''
        # Every 10 frames, add new points at least 10 px from those already tracked.
            if frame_count % 10 == 0:
                mask = np.full(gray2.shape, 255, np.uint8)
                for track in tracks:
                    cv2.circle(mask, np.int32(track[-1]), 10, 0, -1)
                corners = cv2.goodFeaturesToTrack(gray2, maxCorners=300, qualityLevel=0.01,
                                                  minDistance=10, blockSize=7, mask=mask)
                tracks += [[point] for point in corners[:, 0]]
        '''
        # tweak parameters here too instead of every 10 frames, 3 gives a better score
        if (i % 3 == 0):
            mask = np.full(grey2.shape, 255, np.uint8)
            for track in tracks:
                cv2.circle(mask, tuple(int(v) for v in track[-1]), 10, 0, -1)
            corners = cv2.goodFeaturesToTrack(grey2, maxCorners=300, qualityLevel=0.01, minDistance=10, blockSize=7, mask=mask)
            if corners is not None:
                tracks += [[p] for p in corners[:, 0]]

        # at this point we should have tracks filled with all shifts in movement from the animal
        # and we should put a box around those points

        # we can track all the points in tracks that moved 
        # for this part i googled this: in a tracked array of frames cv2 optical flow, how do we find the ones that moved
        '''
        displacement = np.linalg.norm(good_new - good_old, axis=1) # Set a minimum pixel threshold to ignore camera 
        jitter or small noise threshold = 2.0  # in pixels
        moved_points = good_new[displacement > threshold]
        '''
        threshold = 2.0   
        moving = [t[-1] for t in tracks if len(t) >= 2 and np.linalg.norm(np.subtract(t[-1], t[-2])) > threshold]

        # box the moving points, tweaked from >= 5 and got better score, 
        if len(moving) >= 3:
            x_moving = [p[0] for p in moving]
            y_moving = [p[1] for p in moving]
            # make a box with the min and max values
            x0 = min(x_moving)
            x1 = max(x_moving)
            y0 = min(y_moving)
            y1 = max(y_moving)

            # clip to 1280 x 685 size per assignment requirment
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, 1280)
            y1 = min(y1, 685)
            prev_box = (x0, y0, x1 - x0, y1 - y0)

        # update previous box
        if prev_box is not None:
            boxes[i] = prev_box

        grey1 = grey2

    return boxes
