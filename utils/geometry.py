import torch


def make_coord_grid(shape, range=(0, 1), device='cpu', batch_size=None):
    """
        Args:
            shape: (s_1, ..., s_k), grid shape
            range: range for each axis, list or tuple, [minv, maxv] or [[minv_1, maxv_1], ..., [minv_k, maxv_k]]
        Returns:
            (s_1, ..., s_k, k), coordinate grid
    """
    p_lst = []
    for i, n in enumerate(shape):
        p = (torch.arange(n, device=device) + 0.5) / n
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        p = minv + (maxv - minv) * p
        p_lst.append(p)
    coord = torch.stack(torch.meshgrid(*p_lst, indexing='ij'), dim=-1)
    
    if batch_size is not None:
        coord = coord.unsqueeze(0).expand(batch_size, *([-1] * coord.dim()))
    return coord


def make_coord_scale_grid(shape, range=(0, 1), device='cpu', batch_size=None):
    coord = make_coord_grid(shape, range=range, device=device, batch_size=batch_size)
    scale = torch.ones_like(coord)
    for i, n in enumerate(shape):
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        scale[..., i] *= (maxv - minv) / n
    return coord, scale
