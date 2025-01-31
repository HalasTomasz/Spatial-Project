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

        print(len(self.img_flist))

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        fname = self.img_flist[index]
        img_path = os.path.join(self.opt.data_root, fname)

        # Load and process image
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        lbp_mask = self._load_lbp(copy.deepcopy(original_img))
        lbp_mask = transforms.ToTensor()(lbp_mask)

        # Apply transformations
        original_img = self.transform(original_img)

       
        mask_path = os.path.join(self.opt.mask_root.global_mask, fname)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask_global = transforms.ToTensor()(mask)
        
        else:
            mask_global = None

        mask_path = os.path.join(self.opt.mask_root.random_sqaure_mask, fname)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask_square = transforms.ToTensor()(mask)
        
        else:
            mask_square = None
        
        mask_path = os.path.join(self.opt.mask_root.random_walk_mask, fname)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            mask = 255 - mask
            mask_walk = transforms.ToTensor()(mask)
        else:
            mask_walk = None

        mask_global = mask_global if mask_global is not None and mask_global.numel() > 0 else None
        mask_square = mask_square if mask_square is not None and mask_square.numel() > 0 else None
        mask_walk = mask_walk if mask_walk is not None and mask_walk.numel() > 0 else None
        mask_choices = [m for m in [mask_global, mask_square, mask_walk] if m is not None]
        selected_mask = random.choice(mask_choices)

        return {'img': original_img, 
                'mask': selected_mask, 
                'lbp_mask': lbp_mask, 
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