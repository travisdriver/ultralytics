import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ultralytics import YOLO


def parse_line(line: str):
    """
    Expected format per line:
      <class_id> x1 y1 x2 y2 x3 y3 x4 y4
    Returns (class_id, [x1,y1,...,y4]) with floats.
    Ignores empty lines and lines starting with '#'.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.replace(",", " ").split()
    if len(parts) != 9:
        raise ValueError(f"Line has {len(parts)} fields (need 9): {line}")
    cls = int(float(parts[0]))  # allow '10.0' style IDs
    coords = [float(v) for v in parts[1:]]
    return cls, coords


model = YOLO("runs/obb/train2/weights/best.pt")

# from ndarray
im2 = cv2.imread("/home/tdriver6/Documents/ultralytics/datasets/random_poses_0.2ms/images/val/1696123080000000_nav_cam.png")
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
h, w = im2.shape[:2]
results = model.predict(source=im2, save=False, save_txt=False)  # save predictions as labels

xywhrs = results[0].obb.xywhr.cpu().numpy()
conf = results[0].obb.conf.cpu().numpy()
# apply confidence threshold but keep matching confidences for colormap mapping
mask = conf > 0.7
xywhrs = xywhrs[mask]
conf = conf[mask]
print(f"Detected {len(xywhrs)} craters.")

plt.imshow(im2, cmap='gray')

labels_txt = "/home/tdriver6/Documents/ultralytics/datasets/random_poses_0.2ms/labels/val/1696123080000000_nav_cam.txt"
with open(labels_txt, "r") as f:
    for ln, line in enumerate(f, 1):
        cls, coords = parse_line(line)
        coords = np.array(coords).reshape(4, 2)
        coords[:, 0] *= w  # scale x by image width
        coords[:, 1] *= h  # scale y by image height
        polygon = patches.Polygon(coords, closed=True, linewidth=1, edgecolor='green', facecolor='none')
        plt.gca().add_patch(polygon)

# normalize confidences to [0,1] for colormap
import numpy as np
from matplotlib import cm
norm_conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8) if len(conf) > 0 else np.array([])
colormap = cm.get_cmap('jet')
for (xywhr, cval) in zip(xywhrs, conf):
    x, y, w, h, r = xywhr
    color = colormap(cval)
    # Create an Ellipse patch colored by confidence (edge only)
    rect = patches.Ellipse((x, y), w, h, angle=r, linewidth=1, edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)
plt.axis('off')
plt.savefig('test2.png', bbox_inches='tight', pad_inches=0, dpi=300)