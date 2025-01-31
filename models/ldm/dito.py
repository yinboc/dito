import copy
import math

import torch

import models
from omegaconf import OmegaConf
from models import register
from models.ldm.ldm_base import LDMBase
from models.ldm.vqgan.lpips import LPIPS


@register('dito')
class DiTo(LDMBase):

    def __init__(self, render_diffusion, render_sampler, render_n_steps, renderer_guidance=1, lpips=False, **kwargs):
        super().__init__(**kwargs)
        self.render_diffusion = models.make(render_diffusion)
        
        if OmegaConf.is_config(render_sampler):
            render_sampler = OmegaConf.to_container(render_sampler, resolve=True)
        render_sampler = copy.deepcopy(render_sampler)
        if render_sampler.get('args') is None:
            render_sampler['args'] = {}
        render_sampler['args']['diffusion'] = self.render_diffusion
        self.render_sampler = models.make(render_sampler)
        self.render_n_steps = render_n_steps
        self.renderer_guidance = renderer_guidance

        self.t_loss_monitor_v = [0 for _ in range(10)]
        self.t_loss_monitor_n = [0 for _ in range(10)]
        self.t_loss_monitor_decay = 0.99

        self.use_lpips = lpips
        if lpips:
            self.lpips_loss = LPIPS().eval()

    def render(self, z_dec, coord, scale):
        shape = (coord.size(0), 3, coord.size(2), coord.size(3))
        net_kwargs = {'coord': coord, 'scale': scale, 'z_dec': z_dec}
        
        if self.use_ema_renderer:
            self.swap_ema_renderer()
        
        if self.renderer_guidance > 1:
            uncond_z_dec = self.drop_z_emb.unsqueeze(0).expand(z_dec.shape[0], -1, -1, -1)
            uncond_net_kwargs = {'coord': coord, 'scale': scale, 'z_dec': uncond_z_dec}
        else:
            uncond_net_kwargs = None
        
        ret = self.render_sampler.sample(
            net=self.renderer,
            shape=shape,
            n_steps=self.render_n_steps,
            net_kwargs=net_kwargs,
            uncond_net_kwargs=uncond_net_kwargs,
            guidance=self.renderer_guidance,
        )

        if self.use_ema_renderer:
            self.swap_ema_renderer()
        
        return ret

    def forward(self, data, mode, has_optimizer=None):
        if mode in ['z', 'z_dec']:
            ret_z, _ = super().forward(data, mode=mode, has_optimizer=has_optimizer)
            return ret_z
        
        grad = self.get_grad_plan(has_optimizer)
        loss_config = self.loss_config

        if mode == 'pred':
            z_dec, ret = super().forward(data, mode='z_dec', has_optimizer=has_optimizer)

            gt_patch = data['gt'][:, :3, ...]
            coord = data['gt'][:, 3:5, ...]
            scale = data['gt'][:, 5:7, ...]
            
            if grad['renderer']:
                return self.render(z_dec, coord, scale)
            else:
                with torch.no_grad():
                    return self.render(z_dec, coord, scale)

        elif mode == 'loss':
            if not grad['renderer']: # Only training zdm
                _, ret = super().forward(data, mode='z', has_optimizer=has_optimizer)
                return ret
            
            gt_patch = data['gt'][:, :3, ...]
            coord = data['gt'][:, 3:5, ...]
            scale = data['gt'][:, 5:7, ...]

            z_dec, ret = super().forward(data, mode='z_dec', has_optimizer=has_optimizer)
            net_kwargs = {'z_dec': z_dec}

            t = torch.rand(gt_patch.shape[0], device=gt_patch.device)

            if self.gt_noise_lb is not None:
                tmin = torch.ones_like(t) * self.gt_noise_lb
                tmax = torch.ones_like(t) * 1
                t = tmin + (tmax - tmin) * torch.rand_like(tmin)

            if (self.zaug_p is not None) and self.training:
                tz = self._tz
                mask_aug = self._mask_aug
                
                typ = self.zaug_decoding_loss_type
                if typ == 'all':
                    tmin = torch.ones_like(tz) * 0
                    tmax = torch.ones_like(tz) * 1
                elif typ == 'suffix':
                    tmin = tz
                    tmax = torch.ones_like(tz) * 1
                elif typ == 'tz':
                    tmin = tz
                    tmax = tz
                elif typ == 'tmax':
                    tmin = torch.ones_like(tz) * 1
                    tmax = torch.ones_like(tz) * 1
                else:
                    raise NotImplementedError
                t_aug = tmin + (tmax - tmin) * torch.rand_like(tmin)

                t = mask_aug * t_aug + (1 - mask_aug) * t
            
            if not self.use_lpips:
                loss, t = self.render_diffusion.loss(
                    net=self.renderer,
                    x=gt_patch,
                    t=t,
                    net_kwargs=net_kwargs,
                    return_loss_unreduced=True
                )
            else:
                loss, t, x_t, pred = self.render_diffusion.loss(
                    net=self.renderer,
                    x=gt_patch,
                    t=t,
                    net_kwargs=net_kwargs,
                    return_loss_unreduced=True,
                    return_all=True
                )

                sample_pred = x_t + t.view(-1, 1, 1, 1) * pred
                lpips_loss = self.lpips_loss(sample_pred, gt_patch).mean()
                ret['lpips_loss'] = lpips_loss.item()
                lpips_loss_w = loss_config.get('lpips_loss', 1)
                ret['loss'] = ret['loss'] + lpips_loss * lpips_loss_w
            
            # Visualize diffusion network loss for different timesteps #
            if self.training:
                m = len(self.t_loss_monitor_v)
                for i in range(len(loss)):
                    q = min(math.floor(t[i].item() * m), m - 1)
                    self.t_loss_monitor_v[q] = self.t_loss_monitor_v[q] * self.t_loss_monitor_decay + loss[i].item() * (1 - self.t_loss_monitor_decay)
                    self.t_loss_monitor_n[q] += 1
                for q in range(m):
                    if self.t_loss_monitor_n[q] > 0:
                        if self.t_loss_monitor_n[q] < 500:
                            r = 1 - math.pow(self.t_loss_monitor_decay, self.t_loss_monitor_n[q])
                        else:
                            r = 1
                        ret[f'_loss_t{q}'] = self.t_loss_monitor_v[q] / r
            # - #
            
            dae_loss = loss.mean()
            
            ret['dae_loss'] = dae_loss.item()
            dae_loss_w = loss_config.get('dae_loss', 1)
            ret['loss'] = ret['loss'] + dae_loss * dae_loss_w
            return ret
