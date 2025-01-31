import copy
import math

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import models
from models.ldm.vqgan.quantizer import VectorQuantizer


class LDMBase(nn.Module):

    def __init__(
        self,
        encoder,
        z_shape,
        decoder,
        renderer,
        encoder_ema_rate=None,
        decoder_ema_rate=None,
        renderer_ema_rate=None,
        z_gaussian=False,
        z_gaussian_sample=True,
        z_quantizer=False,
        z_quantizer_n_embed=8192,
        z_quantizer_beta=0.25,
        z_layernorm=False,
        zaug_p=None,
        zaug_tmax=1.0,
        zaug_tmax_always=False,
        zaug_decoding_loss_type='all',
        zaug_zdm_diffusion=None,
        gt_noise_lb=None,
        drop_z_p=0.0,
        zdm_net=None,
        zdm_diffusion=None,
        zdm_sampler=None,
        zdm_n_steps=None,
        zdm_ema_rate=0.9999,
        zdm_train_normalize=False,
        zdm_class_cond=None,
        zdm_force_guidance=None,
        loss_config=None,
        use_ema_encoder=False,
        use_ema_decoder=False,
        use_ema_renderer=False,
    ):
        super().__init__()
        self.loss_config = loss_config if loss_config is not None else dict()
        
        self.encoder = models.make(encoder)
        self.decoder = models.make(decoder)
        self.renderer = models.make(renderer)
        
        self.z_shape = tuple(z_shape)

        self.z_gaussian = z_gaussian
        self.z_gaussian_sample = z_gaussian_sample
        
        self.z_quantizer = VectorQuantizer(
            z_quantizer_n_embed,
            z_shape[0],
            beta=z_quantizer_beta,
            remap=None,
            sane_index_shape=False
        ) if z_quantizer else None

        self.z_layernorm = nn.LayerNorm(
            list(z_shape),
            elementwise_affine=False
        ) if z_layernorm else None

        self.zaug_p = zaug_p
        self.zaug_tmax = zaug_tmax
        self.zaug_tmax_always = zaug_tmax_always
        self.zaug_decoding_loss_type = zaug_decoding_loss_type
        if zaug_zdm_diffusion is not None:
            self.zaug_zdm_diffusion = models.make(zaug_zdm_diffusion)

        self.drop_z_p = drop_z_p
        if self.drop_z_p > 0:
            self.drop_z_emb = nn.Parameter(torch.zeros(z_shape[0], z_shape[1], z_shape[2]), requires_grad=False)
        
        self.gt_noise_lb = gt_noise_lb

        # EMA models #
        self.encoder_ema_rate = encoder_ema_rate
        if self.encoder_ema_rate is not None:
            self.encoder_ema = copy.deepcopy(self.encoder)
            for p in self.encoder_ema.parameters():
                p.requires_grad = False
        
        self.decoder_ema_rate = decoder_ema_rate
        if self.decoder_ema_rate is not None:
            self.decoder_ema = copy.deepcopy(self.decoder)
            for p in self.decoder_ema.parameters():
                p.requires_grad = False
        
        self.renderer_ema_rate = renderer_ema_rate
        if self.renderer_ema_rate is not None:
            self.renderer_ema = copy.deepcopy(self.renderer)
            for p in self.renderer_ema.parameters():
                p.requires_grad = False
        # - #

        # z DM #
        if zdm_diffusion is not None:
            self.zdm_diffusion = models.make(zdm_diffusion)
            
            if OmegaConf.is_config(zdm_sampler):
                zdm_sampler = OmegaConf.to_container(zdm_sampler, resolve=True)
            zdm_sampler = copy.deepcopy(zdm_sampler)
            if zdm_sampler.get('args') is None:
                zdm_sampler['args'] = {}
            zdm_sampler['args']['diffusion'] = self.zdm_diffusion
            self.zdm_sampler = models.make(zdm_sampler)
            self.zdm_n_steps = zdm_n_steps

            self.zdm_net = models.make(zdm_net)
            
            self.zdm_net_ema = copy.deepcopy(self.zdm_net)
            for p in self.zdm_net_ema.parameters():
                p.requires_grad = False
            self.zdm_ema_rate = zdm_ema_rate
            
            self.zdm_class_cond = zdm_class_cond

            self.zdm_force_guidance = zdm_force_guidance
        else:
            self.zdm_diffusion = None

        self.zdm_train_normalize = zdm_train_normalize
        if zdm_train_normalize:
            self.register_buffer('zdm_Ez_v', torch.tensor(0.))
            self.register_buffer('zdm_Ez_n', torch.tensor(0.))
            self.register_buffer('zdm_Ez2_v', torch.tensor(0.))
            self.register_buffer('zdm_Ez2_n', torch.tensor(0.))
        # - #

        self.use_ema_encoder = use_ema_encoder
        self.use_ema_decoder = use_ema_decoder
        self.use_ema_renderer = use_ema_renderer

    def get_parameters(self, name):
        if name == 'encoder':
            return self.encoder.parameters()
        elif name == 'decoder':
            p = list(self.decoder.parameters())
            if self.z_quantizer is not None:
                p += list(self.z_quantizer.parameters())
            return p
        elif name == 'renderer':
            return self.renderer.parameters()
        elif name == 'zdm':
            return self.zdm_net.parameters()

    def encode(self, x, return_loss=False, ret=None):
        if self.use_ema_encoder:
            self.swap_ema_encoder()
        
        z = self.encoder(x)

        if self.use_ema_encoder:
            self.swap_ema_encoder()

        if self.z_gaussian:
            posterior = DiagonalGaussianDistribution(z)
            if self.z_gaussian_sample:
                z = posterior.sample()
            else:
                z = posterior.mode()
            kl_loss = posterior.kl().mean()

            if ret is not None:
                ret['z_gau_mean_abs'] = posterior.mean.abs().mean().item()
                ret['z_gau_std'] = posterior.std.mean().item()
        else:
            kl_loss = None
        
        if self.z_layernorm is not None:
            z = self.z_layernorm(z)
        
        if (self.zaug_p is not None) and self.training:
            assert self.z_layernorm is not None # ensure 0 mean 1 std
            if self.zaug_tmax_always:
                tz = torch.ones(z.shape[0], device=z.device) * self.zaug_tmax
            else:
                tz = torch.rand(z.shape[0], device=z.device) * self.zaug_tmax
            zt, _ = self.zaug_zdm_diffusion.add_noise(z, tz)
            mask_aug = (torch.rand(z.shape[0], device=z.device) < self.zaug_p).float()
            z = mask_aug.view(-1, 1, 1, 1) * zt + (1 - mask_aug).view(-1, 1, 1, 1) * z
            self._tz = tz
            self._mask_aug = mask_aug

        if return_loss:
            return z, kl_loss
        else:
            return z

    def decode(self, z, return_loss=False):
        if self.z_quantizer is not None:
            z, quant_loss, _ = self.z_quantizer(z)
        else:
            quant_loss = None

        if self.use_ema_decoder:
            self.swap_ema_decoder()
        
        z_dec = self.decoder(z)

        if self.use_ema_decoder:
            self.swap_ema_decoder()

        if return_loss:
            return z_dec, quant_loss
        else:
            return z_dec
    
    def render(self, z_dec, coord, cell):
        raise NotImplementedError
    
    def normalize_for_zdm(self, z):
        if self.zdm_train_normalize:
            mean = self.zdm_Ez_v
            var = self.zdm_Ez2_v - mean ** 2
            return (z - mean) / torch.sqrt(var)
        else:
            return z
    
    def denormalize_for_zdm(self, z):
        if self.zdm_train_normalize:
            mean = self.zdm_Ez_v
            var = self.zdm_Ez2_v - mean ** 2
            return z * torch.sqrt(var) + mean
        else:
            return z

    def forward(self, data, mode, has_optimizer=None):
        grad = self.get_grad_plan(has_optimizer)
        loss = torch.tensor(0., device=data['inp'].device)
        loss_config = self.loss_config
        ret = dict()

        # Encoder
        if grad['encoder']:
            z, kl_loss = self.encode(data['inp'], return_loss=True, ret=ret)

            if self.z_gaussian:
                ret['kl_loss'] = kl_loss.item()
                loss = loss + kl_loss * loss_config.get('kl_loss', 0.0)
        else:
            with torch.no_grad():
                z, kl_loss = self.encode(data['inp'], return_loss=True, ret=ret)
        
        if self.training and self.drop_z_p > 0:
            drop_mask = (torch.rand(z.shape[0], device=z.device) < self.drop_z_p).to(z.dtype)
            z = drop_mask.view(-1, 1, 1, 1) * self.drop_z_emb.unsqueeze(0) + (1 - drop_mask).view(-1, 1, 1, 1) * z

        # Z DM
        if grad['zdm']:
            if self.zdm_train_normalize and self.training:
                self.zdm_Ez_v = (
                    self.zdm_Ez_v * (self.zdm_Ez_n / (self.zdm_Ez_n + 1))
                    + z.mean().item() / (self.zdm_Ez_n + 1)
                )
                self.zdm_Ez_n = self.zdm_Ez_n + 1
                
                self.zdm_Ez2_v = (
                    self.zdm_Ez2_v * (self.zdm_Ez2_n / (self.zdm_Ez2_n + 1))
                    + (z ** 2).mean().item() / (self.zdm_Ez2_n + 1)
                )
                self.zdm_Ez2_n = self.zdm_Ez2_n + 1
                
                ret['normalize_z_mean'] = self.zdm_Ez_v.item()
                ret['normalize_z_std'] = math.sqrt((self.zdm_Ez2_v - self.zdm_Ez_v ** 2).item())

            z_for_dm = self.normalize_for_zdm(z)
            
            net_kwargs = dict()
            if self.zdm_class_cond is not None:
                net_kwargs['class_labels'] = data['class_labels']

            zdm_loss = self.zdm_diffusion.loss(self.zdm_net, z_for_dm, net_kwargs=net_kwargs)
            ret['zdm_loss'] = zdm_loss.item()
            loss = loss + zdm_loss * loss_config.get('zdm_loss', 1.0)
            
            if not self.training:
                ret['zdm_ema_loss'] = self.zdm_diffusion.loss(self.zdm_net_ema, z_for_dm, net_kwargs=net_kwargs).item()

        # Decoder
        if mode == 'z':
            ret_z = z
        elif mode == 'z_dec':
            if grad['decoder']:
                z_dec, quant_loss = self.decode(z, return_loss=True)
            else:
                with torch.no_grad():
                    z_dec, quant_loss = self.decode(z, return_loss=True)
            ret_z = z_dec

            if self.z_quantizer is not None:
                ret['quant_loss'] = quant_loss.item()
                loss = loss + quant_loss * loss_config.get('quant_loss', 1.0)

        ret['loss'] = loss
        return ret_z, ret

    def get_grad_plan(self, has_optimizer):
        if has_optimizer is None:
            has_optimizer = dict()
        grad = dict()
        grad['encoder'] = has_optimizer.get('encoder', False)
        grad['decoder'] = grad['encoder'] or has_optimizer.get('decoder', False)
        grad['renderer'] = grad['decoder'] or has_optimizer.get('renderer', False)
        grad['zdm'] = has_optimizer.get('zdm', False) # not in chain definition
        return grad
    
    def update_ema_fn(self, net_ema, net, rate):
        if rate != 1:
            for ema_p, cur_p in zip(net_ema.parameters(), net.parameters()):
                ema_p.data.lerp_(cur_p.data, 1 - rate)
    
    def update_ema(self):
        if self.encoder_ema_rate is not None:
            self.update_ema_fn(self.encoder_ema, self.encoder, self.encoder_ema_rate)
        if self.decoder_ema_rate is not None:
            self.update_ema_fn(self.decoder_ema, self.decoder, self.decoder_ema_rate)
        if self.renderer_ema_rate is not None:
            self.update_ema_fn(self.renderer_ema, self.renderer, self.renderer_ema_rate)
        if (self.zdm_diffusion is not None) and (self.zdm_ema_rate is not None):
            self.update_ema_fn(self.zdm_net_ema, self.zdm_net, self.zdm_ema_rate)

    def generate_samples(
        self,
        batch_size,
        n_steps,
        net_kwargs=None,
        uncond_net_kwargs=None,
        ema=False,
        guidance=1.0,
        noise=None,
        render_res=(256, 256),
        return_z=False,
    ):
        if self.zdm_force_guidance is not None:
            guidance = self.zdm_force_guidance
        
        shape = (batch_size,) + self.z_shape
        net = self.zdm_net if not ema else self.zdm_net_ema
        
        z = self.zdm_sampler.sample(
            net,
            shape,
            n_steps,
            net_kwargs=net_kwargs,
            uncond_net_kwargs=uncond_net_kwargs,
            guidance=guidance,
            noise=noise,
        )

        if return_z:
            return z

        if (self.zaug_p is not None) and self.zaug_tmax_always:
            tz = torch.ones(z.shape[0], device=z.device) * self.zaug_tmax
            z, _ = self.zaug_zdm_diffusion.add_noise(z, tz)
        
        z = self.denormalize_for_zdm(z)
        z_dec = self.decode(z)

        coord = torch.zeros(batch_size, 2, render_res[0], render_res[1], device=z_dec.device)
        scale = torch.zeros(batch_size, 2, render_res[0], render_res[1], device=z_dec.device)
        return self.render(z_dec, coord, scale)
    
    def swap_ema_encoder(self):
        _ = self.encoder
        self.encoder = self.encoder_ema
        self.encoder_ema = _
    
    def swap_ema_decoder(self):
        _ = self.decoder
        self.decoder = self.decoder_ema
        self.decoder_ema = _
    
    def swap_ema_renderer(self):
        _ = self.renderer
        self.renderer = self.renderer_ema
        self.renderer_ema = _


class DiagonalGaussianDistribution(object):

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2)
                    + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
