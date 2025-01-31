import os
import random
from PIL import Image, ImageFile

from datasets import register
from torch.utils.data import Dataset
from torchvision import transforms


Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.webp')


@register('class_folder')
class ClassFolder(Dataset):

    def __init__(self, root_path, resize=None, square_crop=False, rand_crop=None, rand_flip=False, drop_label_p=0.0, image_only=False):
        folders = [_ for _ in sorted(os.listdir(root_path)) if os.path.isdir(os.path.join(root_path, _))]
        self.files = []
        self.labels = []
        for i, folder in enumerate(folders):
            for file in sorted(os.listdir(os.path.join(root_path, folder))):
                if file.endswith(IMAGE_EXTS):
                    self.files.append(os.path.join(root_path, folder, file))
                    self.labels.append(i)
        
        self.resize = resize
        self.square_crop = square_crop
        self.rand_crop = rand_crop
        self.rand_flip = transforms.RandomHorizontalFlip() if rand_flip else None
        
        self.n_classes = len(folders)
        self.drop_label_p = drop_label_p

        self.image_only = image_only

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.files[idx]).convert('RGB')
            label = self.labels[idx]
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

        if self.drop_label_p > 0.0 and random.random() < self.drop_label_p:
            label = self.n_classes
        
        if self.image_only:
            return image
        else:
            return {
                'image': image,
                'class_labels': label,
            }
