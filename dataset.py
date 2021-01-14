
# Import libraries
import os
import sys
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import albumentations as A
import torchvision.transforms as tra
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import cv2
import random
import tifffile as tiff
import xml.etree.ElementTree as ET

#import generate_masks as gm



def get_transform(train):
    if train:
        transforms = A.Compose([
        #A.CenterCrop(100, 100),
        #A.RandomCrop(80, 80),
        # A.HorizontalFlip(p=0.5),
        A.Cutout(num_holes=1, max_h_size=200, max_w_size=200, always_apply=True)
        # A.Rotate(limit=(-60, 60), p=0.5) # change the rotation limit; range = from (-6,6) to (-60,60)
        # A.VerticalFlip(p=0.5),
        #A.ShiftScaleRotate(rotate_limit=50, p=0.6)
        #A.SmallestMaxSize(max_size=1292, p=1)
        #A.augmentations.transforms.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=5, p=0.5)
        #A.RandomScale(scale_limit=0.2, p=0.5) # commented for the microscopic images
        #A.core.composition.OneOf([A.Resize(2298, 1292, interpolation=1, always_apply=False, p=1), A.Resize(3109, 1748, interpolation=1, always_apply=False, p=1), A.Resize(2568, 1444, interpolation=1, always_apply=False, p=1), A.Resize(2839, 1596, interpolation=1, always_apply=False, p=1)],p=0.5)
        #A.augmentations.transforms.Resize(2298, 1292, interpolation=1, always_apply=False, p=1)
        #A.VerticalFlip(p=0.5),
        #A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms

class FaceDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(self.root)))
        


    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])


        # img = tiff.imread(img_path)

        
        img=cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        flipped=A.HorizontalFlip(p=0.5)(image=img)
        img=flipped["image"]
      

        # print(np.max(img))
        # img=img/np.max(img)
        # img = np.ndarray.astype(img, dtype=np.float32)
        target=tra.ToTensor()(img)
        
        m, s = cv2.meanStdDev(img)

   


        image_id = torch.tensor([idx])
        # suppose all instances are not crowd

        # target = {}
        # target["image_id"] = image_id

        # make sure your img and mask array are in this format before passing into albumentations transforms, img.shape=[H, W, C] and mask.shape= [N, H, W]
        if self.transforms is not None:
            #data={'image': img, 'masks': masks}
            augmented = self.transforms(image=img)
            img=augmented["image"]
        
        # sometimes one microbe instance may get completely cut off from the bounds after augmentation,
      



        # img = np.moveaxis(img, (0,1,2), (1,2,0)) # pytorch wants a different format for the image ([C, H, W]) so run img=np.moveaxis(img, (0,1,2), (1,2,0)) on the augmented image before turning it into a float tensor
        #print(img.shape)
        # img = np.ndarray.astype(img, dtype=np.float32)
        img=tra.ToTensor()(img)
        # img=tra.Normalize(mean=[m[0][0], m[1][0], m[2][0]], std=[s[0][0], s[1][0], s[2][0]])(img)
        # img=tra.Normalize(mean=[0.1788, 0.1788, 0.1788], std=[0.0481, 0.0481, 0.0481])(img)
        # img=tra.Normalize(mean=[0.1925, 0.1925, 0.1925], std=[0.0531, 0.0531, 0.0531])(img)

        # img=tra.Normalize(mean=[0.1037, 0.1037, 0.1037], std=[0.0490, 0.0490, 0.0490])(img)
        # img=tra.Normalize(mean=[9.2711, 9.2711, 9.2711], std=[13.4850, 13.4850, 13.4850])(img)
        # img=tra.Normalize(mean=[4.0257, 4.0257, 4.0257], std=[6.2710, 6.2710, 6.2710])(img)
        # img=tra.Normalize(mean=[9.2711, 9.2711, 9.2711], std=[13.4850, 13.4850, 13.4850])(img)

        



        return img, target

    def __len__(self):
        return len(self.imgs)
