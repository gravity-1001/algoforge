import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

LABEL_MAP = {
    0:0,
    1:1,
    2:2,
    3:3,
    27:4,
    39:5
}

class SegDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        new_mask = np.zeros_like(mask)

        for k,v in LABEL_MAP.items():
            new_mask[mask == k] = v

        mask = new_mask

        # smaller resolution = faster training
        image = cv2.resize(image,(450,450))
        mask = cv2.resize(mask,(450,450),interpolation=cv2.INTER_NEAREST)

        image = torch.tensor(image).permute(2,0,1).float()/255
        mask = torch.tensor(mask).long()

        return image, mask