import torch
import numpy as np
import random
from torchvision.transforms.functional import resize
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

class MaskGenerator:
    def __init__(self, fine_size=256, overlap=10):
        self.fine_size = fine_size
        self.overlap = overlap

    def create_rand_mask(self):
        """Create a square mask with a random position."""
        h, w = self.fine_size, self.fine_size
        mask = np.zeros((h, w))
        maxt = h - self.overlap - h // 2
        maxl = w - self.overlap - w // 2
        rand_t = np.random.randint(self.overlap, maxt)
        rand_l = np.random.randint(self.overlap, maxl)

        mask[rand_t:rand_t + self.fine_size // 2 - 2 * self.overlap, 
             rand_l:rand_l + self.fine_size // 2 - 2 * self.overlap] = 1

        return torch.ByteTensor(mask), rand_t, rand_l

    def random_walk(self, canvas, ini_x, ini_y, length):
        """Generate a mask by random walk on a canvas."""
        action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        x, y = ini_x, ini_y
        img_size = canvas.shape[-1]
        x_list, y_list = [], []

        for _ in range(length):
            r = random.choice(range(len(action_list)))
            x = np.clip(x + action_list[r][0], 0, img_size - 1)
            y = np.clip(y + action_list[r][1], 0, img_size - 1)
            x_list.append(x)
            y_list.append(y)

        canvas[np.array(x_list), np.array(y_list)] = 0
        return canvas

    def create_random_walk_mask(self):
        """Create a mask using random walk."""
        canvas = np.ones((self.fine_size, self.fine_size), dtype="i")
        ini_x = random.randint(0, self.fine_size - 1)
        ini_y = random.randint(0, self.fine_size - 1)
        return self.random_walk(canvas, ini_x, ini_y, 128 ** 2)

    def wrapper_gmask(self, res=0.06, density=0.4, max_size=300, max_partition=50):
        """Generate a global mask using interpolation and randomization."""
        low_pattern = torch.rand(1, 1, int(res * max_size), int(res * max_size)).mul(255)
        pattern = F.interpolate(low_pattern, (max_size, max_size), mode='bilinear').detach()
        pattern.div_(255)
        pattern = torch.lt(pattern, density).byte()
        pattern = torch.squeeze(pattern).byte()

        return self.create_gmask(pattern, max_size, self.fine_size, max_partition)

    def create_gmask(self, pattern, max_size, fine_size, max_partition, limit_cnt=25):
        """Create a mask from a global pattern."""
        for _ in range(limit_cnt):
            x = random.randint(1, max_size - fine_size)
            y = random.randint(1, max_size - fine_size)
            mask = pattern[y:y + fine_size, x:x + fine_size]
            area = mask.sum() * 100. / (fine_size * fine_size)
            if 10 < area < max_partition:
                return 1 - mask.expand(mask.size(0), mask.size(1))
        return None

    def save_image(self, image, image_path):
        """Save a numpy image array as an image file."""
        if isinstance(image, torch.Tensor):
            image_numpy = image.cpu().numpy()

            binary_image = (image_numpy >= 0.5).astype(np.uint8) * 255
        
        else:
            binary_image = (image >= 0.5).astype(np.uint8) * 255

        image_pil = Image.fromarray(binary_image)
        image_pil.save(image_path)

# if __name__ == "__main__":
#     generator = MaskGenerator(fine_size=256, overlap=10)

#     # Create a random mask
#     rand_mask, rand_t, rand_l = generator.create_rand_mask()
#     print(f"Random Mask:", rand_mask.shape, rand_t, rand_l)
#     plt.imshow(rand_mask, cmap='gray')
#     plt.show()

#     # Create a random walk mask
#     random_walk_mask = generator.create_random_walk_mask()
#     print(f"Random Walk Mask Shape: {random_walk_mask.shape}")
#     plt.imshow(random_walk_mask, cmap='gray')
#     plt.show()

#     # Create a global mask
#     gmask = generator.wrapper_gmask()
#     print(f"Global Mask Shape: {gmask.shape}")
#     plt.imshow(gmask, cmap='gray')
#     plt.show()
