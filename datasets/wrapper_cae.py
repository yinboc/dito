import random
from PIL import Image

import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms

import datasets
from datasets import register
from utils.geometry import make_coord_scale_grid


class BaseWrapperCAE:

    def __init__(
        self,
        dataset,
        resize_inp,
        return_gt=True,
        gt_glores_lb=None,
        gt_glores_ub=None,
        gt_patch_size=None,
        p_whole=0.0,
        p_max=0.0
    ):
        self.dataset = datasets.make(dataset)
        self.resize_inp = resize_inp
        self.return_gt = return_gt
        self.gt_glores_lb = gt_glores_lb
        self.gt_glores_ub = gt_glores_ub
        self.gt_patch_size = gt_patch_size
        self.p_whole = p_whole
        self.p_max = p_max
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def process(self, image):
        assert image.size[0] == image.size[1]
        ret = {}
        
        inp = image.resize((self.resize_inp, self.resize_inp), Image.LANCZOS)
        inp = self.transform(inp)
        ret.update({'inp': inp})
        if not self.return_gt:
            return ret

        if self.gt_glores_lb is None:
            glo = self.transform(image)
        else:
            if random.random() < self.p_whole:
                r = self.gt_patch_size
            elif random.random() < self.p_max:
                r = min(image.size[0], self.gt_glores_ub)
            else:
                r = random.randint(
                    self.gt_glores_lb,
                    max(self.gt_glores_lb, min(image.size[0], self.gt_glores_ub))
                )
            glo = image.resize((r, r), Image.LANCZOS)
            glo = self.transform(glo)

        p = self.gt_patch_size
        ii = random.randint(0, glo.shape[1] - p)
        jj = random.randint(0, glo.shape[2] - p)
        gt_patch = glo[:, ii: ii + p, jj: jj + p]

        x0, y0 = ii / glo.shape[-2], jj / glo.shape[-1]
        x1, y1 = (ii + p) / glo.shape[-2], (jj + p) / glo.shape[-1]
        coord, scale = make_coord_scale_grid((p, p), range=[[x0, x1], [y0, y1]])
        ret['gt'] = torch.cat([
            gt_patch, # 3 p p
            coord.permute(2, 0, 1), # 2 p p
            scale.permute(2, 0, 1), # 2 p p
        ], dim=0)

        return ret


@register('wrapper_cae')
class WrapperCAE(BaseWrapperCAE, Dataset):
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, dict):
            ret = dict()
            ret.update(self.process(data.pop('image')))
            ret.update(data)
            return ret
        else:
            return self.process(data)


@register('wrapper_cae_iterable')
class WrapperCAE(BaseWrapperCAE, IterableDataset):

    def __iter__(self):
        for data in self.dataset:
            if isinstance(data, dict):
                ret = dict()
                ret.update(self.process(data.pop('image')))
                ret.update(data)
                yield ret
            else:
                yield self.process(data)
