import cv2
import numpy as np

'''
def flow_to_bgr(flow, max_flow=5):
    # Hue: direction. Saturation: magnitude. White: no motion.
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = angle / 2
    hsv[..., 1] = np.clip(255 * magnitude / max_flow, 0, 255)
    hsv[..., 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

video = cv2.VideoCapture("raccoon_reference.mp4")
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

_, previous = video.read()
gray1 = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

# adding some code to write back to same directory
height, width = previous.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# farneback_writer = cv2.VideoWriter(
#     "farneback_output.mp4",
#     fourcc,
#     10,
#     (width, height)
# )

dis_writer = cv2.VideoWriter(
    "dis_output.mp4",
    fourcc,
    10,
    (width, height)
)
# done 

while True:
    ok, current = video.read()
    if not ok:
        break
    gray2 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    # farneback = cv2.calcOpticalFlowFarneback(
    #     gray1, gray2, None, pyr_scale=0.5, levels=3, winsize=15,
    #     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    dis_flow = dis.calc(gray1, gray2, None)

    mag = np.linalg.norm(dis_flow, axis=2)
    THRESH = 1.0   # adjust from stage 1
    mask = (mag > THRESH).astype(np.uint8) * 255
    cv2.imshow("Mask", mask)

    # added this code for writeback
    # farneback_img = flow_to_bgr(farneback)
    dis_img = flow_to_bgr(dis_flow)

    # farneback_writer.write(farneback_img)
    dis_writer.write(dis_img)
    # done

    # Forward flow is defined at pixels in the previous frame.
    cv2.imshow("Video", previous)
    # cv2.imshow("Farneback", flow_to_bgr(farneback))
    cv2.imshow("DIS", flow_to_bgr(dis_flow))

    previous, gray1 = current, gray2
    if cv2.waitKey(100) == 27:  # Esc quits.
        break

# add code
#farneback_writer.release()

dis_writer.release()
# done

video.release()
cv2.destroyAllWindows()
'''

# the goal is to take what i did for the example and make it into frames and track animal
# reference: https://medium.com/@itberrios6/introduction-to-motion-detection-part-2-6ec3d6b385d4

# create a bunch of helper functions so main workflow makes sense
def preprocess(frame):
    # convert to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grey

def get_motion_mask(flow, thresh=1, kernel=np.ones((7, 7), np.uint8)):
    # this function will get detection mask from optical flow
    mag = np.linalg.norm(flow, axis=2)
    motion_mask = np.uint8(mag > thresh) * 255
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return motion_mask

def get_bounding_box(motion_mask):
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, # cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
        # get area of contour
        # x,y,w,h = cv2.boundingRect(cnt)
        # area = w*h
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # pick an area that can be the minimum accepted, this can be tweaked after implementation
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h, area])

    if not detections:
        return None

    # keep the detection with the largest area
    best = max(detections, key=lambda d: d[4])
    x, y, w, h = best[:4]
    # Each image is 1280 pixels wide and 685 pixels high
    # clamp to 1280 x 685
    w = min(max(w, 1), 1280)
    h = min(max(h, 1), 685)
    x = min(max(x, 0), 1280 - w)
    y = min(max(y, 0), 685 - h)
    return (x, y, w, h)

def track_animal(frames):
    # now with the video class DIS example, I am going to try and recreate it for frames
    boxes = np.full((len(frames), 4), np.nan, dtype=float)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # get your original frame, but for the purpose of looping get "prev frame"
    prev_grey = preprocess(frames[0])
    # previous box is none right now, no bounding box created
    prev_box = None

    # replaces the while true, while there are frames do this
    for i in range(1, len(frames)):
        # process the current frame
        curr_grey = preprocess(frames[i])
        # dis_flow = dis.calc(gray1, gray2, None)
        dis_flow = dis.calc(prev_grey, curr_grey, None)
        # get the mask and pass through our boundng box helper
        flow_mask = get_motion_mask(dis_flow)
        detected = get_bounding_box(flow_mask)
        # if not none then make prev box the detected box for next
        if detected is not None:
            prev_box = detected
        # put the box into boxes
        if prev_box is not None:
            boxes[i] = prev_box
        # move onto next
        prev_grey = curr_grey
    return boxes