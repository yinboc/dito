import torch


models = dict()


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def load_sd_from_ckpt(ckpt, keys_only=None):
    sd = torch.load(ckpt, map_location='cpu')['model']['sd']
    if keys_only is not None:
        keys_only_dot = tuple([_ + '.' for _ in keys_only])
        keys_only = set(keys_only)
        for k in list(sd.keys()):
            if not (k in keys_only or k.startswith(keys_only_dot)):
                sd.pop(k)
    return sd


def make(spec, load_sd=False):
    args = spec.get('args')
    if args is None:
        args = dict()
    model = models[spec['name']](**args)

    if spec.get('load_ckpt') is not None:
        sd = load_sd_from_ckpt(spec['load_ckpt'], spec.get('load_ckpt_keys_only'))
        model.load_state_dict(sd, strict=False)
    
    if load_sd:
        model.load_state_dict(spec['sd'])

    return model


@register('identity')
def make_identity():
    return torch.nn.Identity()
