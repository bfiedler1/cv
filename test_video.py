import cv2
import numpy as np

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