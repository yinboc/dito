# https://gist.github.com/mrsteyk/74ad3ec2f6f823111ae4c90e168505ac

import torch
import torch.nn.functional as F
import torch.nn as nn

from models import register


class TimestepEmbedding(nn.Module):
    def __init__(self, n_time=1024, n_emb=320, n_out=1280) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_time, n_emb)
        self.f_1 = nn.Linear(n_emb, n_out)
        self.f_2 = nn.Linear(n_out, n_out)

    def forward(self, x) -> torch.Tensor:
        x = self.emb(x)
        x = self.f_1(x)
        x = F.silu(x)
        return self.f_2(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, pe_dim=320, out_dim=1280, max_positions=10000, endpoint=True):
        super().__init__()
        self.num_channels = pe_dim
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.f_1 = nn.Linear(pe_dim, out_dim)
        self.f_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        
        x = self.f_1(x)
        x = F.silu(x)
        return self.f_2(x)


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels=320) -> None:
        super().__init__()
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(x)


class ImageUnembedding(nn.Module):
    def __init__(self, in_channels=320, out_channels=3) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(32, in_channels)
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(F.silu(self.gn(x)))


class ConvResblock(nn.Module):
    def __init__(self, in_features, out_features, t_dim) -> None:
        super().__init__()
        self.f_t = nn.Linear(t_dim, out_features * 2)

        self.gn_1 = nn.GroupNorm(32, in_features)
        self.f_1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)

        self.gn_2 = nn.GroupNorm(32, out_features)
        self.f_2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1)

        skip_conv = in_features != out_features
        self.f_s = (
            nn.Conv2d(in_features, out_features, kernel_size=1, padding=0)
            if skip_conv
            else nn.Identity()
        )

    def forward(self, x, t):
        x_skip = x
        t = self.f_t(F.silu(t))
        t = t.chunk(2, dim=1)
        t_1 = t[0].unsqueeze(dim=2).unsqueeze(dim=3) + 1
        t_2 = t[1].unsqueeze(dim=2).unsqueeze(dim=3)

        gn_1 = F.silu(self.gn_1(x))
        f_1 = self.f_1(gn_1)

        gn_2 = self.gn_2(f_1)

        return self.f_s(x_skip) + self.f_2(F.silu(gn_2 * t_1 + t_2))


# Also ConvResblock
class Downsample(nn.Module):
    def __init__(self, in_channels, t_dim) -> None:
        super().__init__()
        self.f_t = nn.Linear(t_dim, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2).unsqueeze(3) + 1
        t_2 = t_2.unsqueeze(2).unsqueeze(3)

        gn_1 = F.silu(self.gn_1(x))
        avg_pool2d = F.avg_pool2d(gn_1, kernel_size=(2, 2), stride=None)
        f_1 = self.f_1(avg_pool2d)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.avg_pool2d(x_skip, kernel_size=(2, 2), stride=None)


# Also ConvResblock
class Upsample(nn.Module):
    def __init__(self, in_channels, t_dim) -> None:
        super().__init__()
        self.f_t = nn.Linear(t_dim, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2).unsqueeze(3) + 1
        t_2 = t_2.unsqueeze(2).unsqueeze(3)

        gn_1 = F.silu(self.gn_1(x))
        upsample = F.upsample_nearest(gn_1, scale_factor=2)
        f_1 = self.f_1(upsample)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.upsample_nearest(x_skip, scale_factor=2)


@register('consistency_decoder_unet')
class ConsistencyDecoderUNet(nn.Module):
    def __init__(self, in_channels=3, z_dec_channels=None, c0=320, c1=640, c2=1024, pe_dim=320, t_dim=1280) -> None:
        super().__init__()
        if z_dec_channels is not None:
            in_channels += z_dec_channels
        self.embed_image = ImageEmbedding(in_channels=in_channels, out_channels=c0)
        self.embed_time = PositionalEmbedding(pe_dim=pe_dim, out_dim=t_dim)

        down_0 = nn.ModuleList([
            ConvResblock(c0, c0, t_dim),
            ConvResblock(c0, c0, t_dim),
            ConvResblock(c0, c0, t_dim),
            Downsample(c0, t_dim),
        ])
        down_1 = nn.ModuleList([
            ConvResblock(c0, c1, t_dim),
            ConvResblock(c1, c1, t_dim),
            ConvResblock(c1, c1, t_dim),
            Downsample(c1, t_dim),
        ])
        down_2 = nn.ModuleList([
            ConvResblock(c1, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
            Downsample(c2, t_dim),
        ])
        down_3 = nn.ModuleList([
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
        ])
        self.down = nn.ModuleList([
            down_0,
            down_1,
            down_2,
            down_3,
        ])

        self.mid = nn.ModuleList([
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
        ])

        up_3 = nn.ModuleList([
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            Upsample(c2, t_dim),
        ])
        up_2 = nn.ModuleList([
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 + c1, c2, t_dim),
            Upsample(c2, t_dim),
        ])
        up_1 = nn.ModuleList([
            ConvResblock(c2 + c1, c1, t_dim),
            ConvResblock(c1 * 2, c1, t_dim),
            ConvResblock(c1 * 2, c1, t_dim),
            ConvResblock(c0 + c1, c1, t_dim),
            Upsample(c1, t_dim),
        ])
        up_0 = nn.ModuleList([
            ConvResblock(c0 + c1, c0, t_dim),
            ConvResblock(c0 * 2, c0, t_dim),
            ConvResblock(c0 * 2, c0, t_dim),
            ConvResblock(c0 * 2, c0, t_dim),
        ])
        self.up = nn.ModuleList([
            up_0,
            up_1,
            up_2,
            up_3,
        ])

        self.output = ImageUnembedding(in_channels=c0)
    
    def get_last_layer_weight(self):
        return self.output.f.weight

    def forward(self, x, t=None, z_dec=None) -> torch.Tensor:
        if z_dec is not None:
            if z_dec.shape[-2] != x.shape[-2] or z_dec.shape[-1] != x.shape[-1]:
                assert x.shape[-2] // z_dec.shape[-2] == x.shape[-1] // z_dec.shape[-1]
                z_dec = F.upsample_nearest(z_dec, scale_factor=x.shape[-2] // z_dec.shape[-2])
            x = torch.cat([x, z_dec], dim=1)
        
        x = self.embed_image(x)

        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)        
        t = self.embed_time(t)

        skips = [x]
        for down in self.down:
            for block in down:
                x = block(x, t)
                skips.append(x)

        for mid in self.mid:
            x = mid(x, t)

        for up in self.up[::-1]:
            for block in up:
                if isinstance(block, ConvResblock):
                    x = torch.concat([x, skips.pop()], dim=1)
                x = block(x, t)

        return self.output(x)
