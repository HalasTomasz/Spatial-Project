import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import copy

class MyDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_flist = sorted(os.listdir(opt.data_root))
        self.mask_flist = sorted(os.listdir(opt.mask_root.global_mask))

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        fname = self.img_flist[index]
        img_path = os.path.join(self.opt.data_root, fname)

        # Load and process image
        orginal_img = cv2.imread(img_path)
        lbp_mask = self._load_lbp(copy.deepcopy(orginal_img))

        # Apply transformations
        orginal_img = self.transform(orginal_img)
        binary_lbp_mask = lbp_mask[0, :, :].unsqueeze(0)  # Shape: (1, H, W)

       
        mask_path = os.path.join(self.opt.mask_root.global_mask, self.mask_flist[index])
        mask = Image.open(mask_path)
        mask_global = transforms.ToTensor()(mask)

        mask_path = os.path.join(self.opt.mask_root.random_sqaure_mask, self.mask_flist[index])
        mask = Image.open(mask_path)
        mask_sqaure = transforms.ToTensor()(mask)

        mask_path = os.path.join(self.opt.mask_root.random_walk_mask, self.mask_flist[index])
        mask = Image.open(mask_path)
        mask_walk = transforms.ToTensor()(mask)
        
        return {'img': orginal_img, 
                'mask_global': mask_global, 
                'mask_sqaure': mask_sqaure, 
                'mask_walk': mask_walk, 
                'lbp_mask': binary_lbp_mask, 
                'pic_name': fname}

    def __len__(self):
        return len(self.img_flist)

    def _get_pixel(self, img, center, x, y):
        try:
            return 1 if img[x][y] >= center else 0
        except IndexError:
            return 0

    def _lbp_calculated_pixel(self, img, x, y):
        '''
         64 | 128 |   1
        ----------------
         32 |   0 |   2
        ----------------
         16 |   8 |   4
        '''
        center = img[x][y]
        offsets = [
            (-1, 1), (0, 1), (1, 1), (1, 0),
            (1, -1), (0, -1), (-1, -1), (-1, 0)
        ]
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]

        val = sum(
            self._get_pixel(img, center, x + dx, y + dy) * p
            for (dx, dy), p in zip(offsets, power_val)
        )
        return val

    def _load_lbp(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_lbp = np.zeros_like(img_gray, dtype=np.uint8)

        for i in range(img_gray.shape[0]):
            for j in range(img_gray.shape[1]):
                img_lbp[i, j] = self._lbp_calculated_pixel(img_gray, i, j)

        return img_lbp