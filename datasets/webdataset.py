import json

import webdataset as wds
from webdataset.handlers import warn_and_continue

from datasets import register


def webdataset_preprocessors(square_crop=True):
    def identity(x):
        if isinstance(x, bytes):
            x = x.decode('utf-8')
        return x
    
    def transform(image):
        w, h = image.size
        l = min(w, h)
        left, upper = (w - l) // 2, (h - l) // 2
        return image.crop((left, upper, left + l, upper + l))

    ret = [
        ('jpg;png', transform if square_crop else lambda x: x, 'image'),
        ('txt', identity, 'caption'),
    ]
    
    return ret


@register('webdataset')
def make_webdataset(json_file, **kwargs):
    with open(json_file, 'r') as file:
        tar_list = json.load(file)
    preprocessors = webdataset_preprocessors(**kwargs)
    handler = warn_and_continue
    dataset = wds.WebDataset(
        tar_list, resampled=True, handler=handler
    ).shuffle(690, handler=handler).decode(
        "pilrgb", handler=handler
    ).to_tuple(
        *[p[0] for p in preprocessors], handler=handler
    ).map_tuple(
        *[p[1] for p in preprocessors], handler=handler
    ).map(lambda x: {p[2]: x[i] for i, p in enumerate(preprocessors)})

    return dataset
