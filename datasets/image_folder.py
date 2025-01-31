import os
import random
from PIL import Image, ImageFile

from datasets import register
from torch.utils.data import Dataset
from torchvision import transforms


Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.webp')


@register('image_folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, resize=None, square_crop=False, rand_crop=None, rand_flip=False):
        files = sorted(os.listdir(root_path))
        self.files = [os.path.join(root_path, _) for _ in files if _.endswith(IMAGE_EXTS)]
        
        self.resize = resize
        self.square_crop = square_crop
        self.rand_crop = rand_crop
        self.rand_flip = transforms.RandomHorizontalFlip() if rand_flip else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.files[idx]).convert('RGB')
        except:
            print('Error loading image:', self.files[idx])
            return self.__getitem__((idx + 1) % self.__len__())
        
        if self.resize is not None:
            r = self.resize
            if isinstance(r, int):
                w, h = image.size
                if w < h:
                    r = (r, int(h / w * r))
                else:
                    r = (int(w / h * r), r)
            image = image.resize(r, Image.LANCZOS)

        if self.square_crop:
            w, h = image.size
            l = min(w, h)
            left, upper = (w - l) // 2, (h - l) // 2
            image = image.crop((left, upper, left + l, upper + l))

        if self.rand_crop is not None:
            w, h = image.size
            left = random.randint(0, w - self.rand_crop)
            upper = random.randint(0, h - self.rand_crop)
            image = image.crop((left, upper, left + self.rand_crop, upper + self.rand_crop))
        
        if self.rand_flip is not None:
            image = self.rand_flip(image)
        
        return image
