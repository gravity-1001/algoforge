import torch
import cv2
import os
import numpy as np
from transformers import SegformerForSemanticSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "segformer_offroad_model.pth"

INPUT_FOLDER = r"C:\Users\uppal\OneDrive\Desktop\offroad-segformer\dataset\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images"
OUTPUT_FOLDER = "predictions"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=6,
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.to(DEVICE)
model.eval()

print("Model loaded")

for img_name in os.listdir(INPUT_FOLDER):

    img_path = os.path.join(INPUT_FOLDER, img_name)

    image = cv2.imread(img_path)

    if image is None:
        print("Skipping bad image:", img_name)
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(rgb, (384,384))

    tensor = torch.tensor(resized).permute(2,0,1).float()/255
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(pixel_values=tensor)

        logits = outputs.logits

        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    pred_resized = cv2.resize(pred,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)

    mask = (pred_resized * 40).astype(np.uint8)

    out_path = os.path.join(OUTPUT_FOLDER,img_name)

    cv2.imwrite(out_path,mask)

    print("Saved:",img_name)

print("Prediction complete")