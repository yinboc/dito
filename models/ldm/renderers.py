import torch.nn as nn

import models
from models import register


@register('fixres_renderer_wrapper')
class FixresRendererWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = models.make(net)
    
    def forward(self, x, coord=None, scale=None, **kwargs):
        return self.net(x, **kwargs)

    def get_last_layer_weight(self):
        return self.net.get_last_layer_weight()
