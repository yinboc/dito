import torch.nn as nn


from models import register
from .model import Encoder, Decoder


default_configs = {
    'f8c4': dict(
        double_z=False,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        give_pre_end=True,
    ),
    'f16c8': dict(
        double_z=False,
        z_channels=8,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        give_pre_end=True,
    ),
}


@register('vqgan_encoder')
def make_vqgan_encoder(config_name, **kwargs):
    encoder_kwargs = default_configs[config_name]
    encoder_kwargs.update(kwargs)
    enc_out_channels = encoder_kwargs['z_channels'] * (2 if encoder_kwargs['double_z'] else 1)
    return nn.Sequential(
        Encoder(**encoder_kwargs),
        nn.Conv2d(enc_out_channels, enc_out_channels, 1),
    )


@register('vqgan_decoder')
def make_vqgan_decoder(config_name, **kwargs):
    decoder_kwargs = default_configs[config_name]
    decoder_kwargs.update(kwargs)
    dec_in_channels = decoder_kwargs['z_channels']
    return nn.Sequential(
        nn.Conv2d(dec_in_channels, dec_in_channels, 1),
        Decoder(**decoder_kwargs),
    )
