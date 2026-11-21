import json
import hw10
from hw10_support import PATTERN_SIZE, load_photos, square_size_mm

paths, images = load_photos("photos")
rms, K, dist, rvecs, tvecs = hw10.calibrate_camera(images, PATTERN_SIZE, square_size_mm(hw10))
h, w = images[0].shape[:2]
with open("calibration.json", "w") as f:
    json.dump({"K": K.tolist(), "dist": dist.ravel().tolist(), "image_size": [w, h]}, f, indent=2)
print("Saved calibration.json")