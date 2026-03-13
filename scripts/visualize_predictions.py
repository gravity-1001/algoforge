import cv2
import os
import numpy as np

IMAGE_FOLDER = r"C:\Users\uppal\OneDrive\Desktop\offroad-segformer\dataset\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images"
PRED_FOLDER = "predictions"
OUTPUT_FOLDER = "overlay_results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for name in os.listdir(PRED_FOLDER):

    pred_path = os.path.join(PRED_FOLDER, name)
    img_path = os.path.join(IMAGE_FOLDER, name)

    pred = cv2.imread(pred_path, 0)
    img = cv2.imread(img_path)

    if img is None or pred is None:
        continue

    pred_color = cv2.applyColorMap((pred*40).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, pred_color, 0.4, 0)

    out_path = os.path.join(OUTPUT_FOLDER, name)

    cv2.imwrite(out_path, overlay)

    print("Overlay saved:", name)

print("Visualization complete")