import os

import torch
import torch.nn.functional as F
import torch.distributed as dist

from models import register
from models.ldm.ldm_base import LDMBase
from models.ldm.vqgan.lpips import LPIPS
from models.ldm.vqgan.discriminator import make_discriminator


@register('glpto')
class GLPTo(LDMBase):

    def __init__(self, lpips=True, disc=True, adaptive_gan_weight=True, noise_render=False, **kwargs):
        super().__init__(**kwargs)
        if lpips:
            self.lpips_loss = LPIPS().eval()
        self.disc = make_discriminator(input_nc=3) if disc else None
        self.adaptive_gan_weight = adaptive_gan_weight
        self.noise_render = noise_render

    def get_parameters(self, name):
        if name == 'disc':
            return self.disc.parameters()
        else:
            return super().get_parameters(name)

    def render(self, z_dec, coord, scale):
        if not self.noise_render:
            return self.renderer(z_dec, coord=coord, scale=scale)
        else:
            shape = (coord.shape[0], 3, coord.shape[2], coord.shape[3])
            noise = torch.randn(shape, device=z_dec.device)
            return self.renderer(noise, coord=coord, scale=scale, z_dec=z_dec)

    def forward(self, data, mode, has_optimizer=None, use_gan=False):
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
            pred = self.render(z_dec, coord, scale)

            l1_loss = torch.abs(pred - gt_patch).mean()
            ret['l1_loss'] = l1_loss.item()
            l1_loss_w = loss_config.get('l1_loss', 1)
            ret['loss'] = ret['loss'] + l1_loss * l1_loss_w

            lpips_loss = self.lpips_loss(pred, gt_patch).mean()
            ret['lpips_loss'] = lpips_loss.item()
            lpips_loss_w = loss_config.get('lpips_loss', 1)
            ret['loss'] = ret['loss'] + lpips_loss * lpips_loss_w

            if use_gan:
                logits_fake = self.disc(pred)
                
                gan_g_loss = -torch.mean(logits_fake)
                ret['gan_g_loss'] = gan_g_loss.item()
                weight = loss_config.get('gan_g_loss', 1)
                
                if self.training and self.adaptive_gan_weight:
                    nll_loss = l1_loss * l1_loss_w + lpips_loss * lpips_loss_w
                    adaptive_gan_w = self.calculate_adaptive_gan_w(nll_loss, gan_g_loss, self.renderer.get_last_layer_weight())
                    ret['adaptive_gan_w'] = adaptive_gan_w.item()
                    weight = weight * adaptive_gan_w
                
                ret['loss'] = ret['loss'] + gan_g_loss * weight

            return ret

        elif mode == 'disc_loss':
            gt_patch = data['gt'][:, :3, ...]
            coord = data['gt'][:, 3:5, ...]
            scale = data['gt'][:, 5:7, ...]
            
            with torch.no_grad():
                z_dec, _ = super().forward(data, mode='z_dec', has_optimizer=None)
                pred = self.render(z_dec, coord, scale)

            logits_real = self.disc(gt_patch)
            logits_fake = self.disc(pred)
            
            disc_loss_type = loss_config.get('disc_loss_type', 'hinge')
            if disc_loss_type == 'hinge':
                loss_real = torch.mean(F.relu(1. - logits_real))
                loss_fake = torch.mean(F.relu(1. + logits_fake))
                loss = (loss_real + loss_fake) / 2
            elif disc_loss_type == 'vanilla':
                loss_real = torch.mean(F.softplus(-logits_real))
                loss_fake = torch.mean(F.softplus(logits_fake))
                loss = (loss_real + loss_fake) / 2
            
            return {
                'loss': loss,
                'disc_logits_real': logits_real.mean().item(),
                'disc_logits_fake': logits_fake.mean().item(),
            }

    def calculate_adaptive_gan_w(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size > 1:
            dist.all_reduce(nll_grads, op=dist.ReduceOp.SUM)
            nll_grads.div_(world_size)
            dist.all_reduce(g_grads, op=dist.ReduceOp.SUM)
            g_grads.div_(world_size)
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
