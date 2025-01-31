import os
import random

import torch
import torch.distributed as dist
import torch_fidelity
import torchvision
from PIL import Image
from torchvision import transforms

import utils
from utils.geometry import make_coord_scale_grid
from .trainers import register
from trainers.base_trainer import BaseTrainer


@register('ldm_trainer')
class LDMTrainer(BaseTrainer):

    def make_model(self):
        super().make_model()
        self.has_optimizer = dict()
        for name, m in self.model.named_children():
            self.log(f'  .{name} {utils.compute_num_params(m)}')

    def make_optimizers(self):
        self.optimizers = dict()
        self.has_optimizer = dict()
        for name, spec in self.config.optimizers.items():
            self.optimizers[name] = utils.make_optimizer(self.model.get_parameters(name), spec)
            self.has_optimizer[name] = True

    def train_step(self, data, bp=True):
        kwargs = {'has_optimizer': self.has_optimizer}

        # Use GAN (GAE) #
        gan_iter = self.config.get('gan_start_after_iters')
        if ((gan_iter is not None) and self.iter > gan_iter):
            kwargs['use_gan'] = True
        # - #

        if self.config.get('autocast_bfloat16', False):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                ret = self.model_ddp(data, mode='loss', **kwargs)
        else:
            ret = self.model_ddp(data, mode='loss', **kwargs)

        loss = ret.pop('loss')
        ret['loss'] = loss.item()
        if bp:
            self.model_ddp.zero_grad()
            loss.backward()
            for name, o in self.optimizers.items():
                if name != 'disc':
                    o.step()

        # Discriminator turn (GAE) #
        if kwargs.get('use_gan', False) == True:
            if self.config.get('autocast_bfloat16', False):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    disc_ret = self.model_ddp(data, mode='disc_loss', **kwargs)
            else:
                disc_ret = self.model_ddp(data, mode='disc_loss', **kwargs)
            
            loss = disc_ret.pop('loss')
            ret['disc_loss'] = loss.item()
            ret.update(disc_ret)
            if bp:
                self.optimizers['disc'].zero_grad()
                loss.backward()
                self.optimizers['disc'].step()
        # - #

        self.model.update_ema()

        return ret
    
    def evaluate(self):
        self.model_ddp.eval()

        ave_scalars = dict()
        pbar = self.loaders['val']

        for data in pbar:
            for k, v in data.items():
                data[k] = v.to(self.device) if torch.is_tensor(v) else v
            
            ret = self.train_step(data, bp=False)

            bs = len(next(iter(data.values())))
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=bs)
        
        self.sync_ave_scalars(ave_scalars)
        
        # Extra evaluation #
        if self.config.get('evaluate_ae', False):
            ave_scalars.update(self.evaluate_ae())
        
        if self.config.get('evaluate_zdm', False):
            ema = self.config.get('evaluate_zdm_ema', True)
            ave_scalars.update(self.evaluate_zdm(ema=ema))
        # - #

        logtext = 'val:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_scalar('val/' + k, v.item())
        self.log_buffer.append(logtext)
        
        return ave_scalars

    def visualize(self):
        self.model_ddp.eval()

        if self.config.get('evaluate_ae', False):
            # self.visualize_ae_fixset()
            self.visualize_ae_random()
        
        if self.config.get('evaluate_zdm', False):
            ema = self.config.get('evaluate_zdm_ema', True)
            # self.visualize_zdm_fixset(ema=ema)
            self.visualize_zdm_random(ema=ema)
            # self.visualize_zdm_denoising(ema=ema)

    def evaluate_ae(self):
        max_samples = self.config.get('eval_ae_max_samples')
        self.loader_samplers['eval_ae'].set_epoch(0)
        
        to_pil = transforms.ToPILImage()
        psnr_value = utils.Averager()
        cnt = 0

        cache_gen_dir = os.path.join(self.env['save_dir'], 'cache', 'fid_gen')
        cache_gt_dir = os.path.join(self.env['save_dir'], 'cache', 'fid_gt')
        if self.is_master:
            utils.ensure_path(cache_gen_dir, force_replace=True)
            utils.ensure_path(cache_gt_dir, force_replace=True)
        dist.barrier()

        for data in self.loaders['eval_ae']:
            for k, v in data.items():
                data[k] = v.to(self.device) if torch.is_tensor(v) else v
            
            pred = self.model(data, mode='pred')
            gt_patch = data['gt'][:, :3, ...]

            pred = (pred * 0.5 + 0.5).clamp(0, 1)
            gt_patch = (gt_patch * 0.5 + 0.5).clamp(0, 1)

            # PSNR
            mse = (pred - gt_patch).pow(2).mean(dim=[1, 2, 3])
            psnr_value.add((-10 * torch.log10(mse)).mean().item())

            # FID
            for i in range(len(pred)):
                idx = int(os.environ['RANK']) + cnt * int(os.environ['WORLD_SIZE'])
                if max_samples is None or idx < max_samples: 
                    to_pil(pred[i]).save(os.path.join(cache_gen_dir, f'{idx}.png'))
                    to_pil(gt_patch[i]).save(os.path.join(cache_gt_dir, f'{idx}.png'))
                cnt += 1
        dist.barrier()

        vt = torch.tensor(psnr_value.item(), device=self.device)
        dist.all_reduce(vt, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        psnr_value = vt.item() / int(os.environ['WORLD_SIZE'])

        if self.is_master:
            metrics = torch_fidelity.calculate_metrics(
                input1=cache_gen_dir,
                input2=cache_gt_dir,
                cuda=True,
                fid=True,
                verbose=False, 
            )
            prefix = 'eval_ae'
            ret = {
                f'{prefix}/PSNR': psnr_value,
                f'{prefix}/FID': metrics['frechet_inception_distance'],
            }
        else:
            ret = {}
        dist.barrier()

        ret = {k: utils.Averager(v) for k, v in ret.items()}
        return ret

    def evaluate_zdm(self, ema):
        max_samples = self.config.get('eval_zdm_max_samples')
        self.loader_samplers['eval_zdm'].set_epoch(0)
        
        to_pil = transforms.ToPILImage()
        cnt = 0

        cache_gen_dir = os.path.join(self.env['save_dir'], 'cache', 'fid_gen')
        cache_gt_dir = os.path.join(self.env['save_dir'], 'cache', 'fid_gt')
        if self.is_master:
            utils.ensure_path(cache_gen_dir, force_replace=True)
            utils.ensure_path(cache_gt_dir, force_replace=True)
        dist.barrier()

        for data in self.loaders['eval_zdm']:
            for k, v in data.items():
                data[k] = v.to(self.device) if torch.is_tensor(v) else v
            
            gt_patch = data['inp']

            net_kwargs = dict()
            uncond_net_kwargs = dict()
            if self.model.zdm_class_cond is not None:
                net_kwargs['class_labels'] = data['class_labels']

                setting = self.config['visualize_zdm_setting']
                uncond_net_kwargs['class_labels'] = setting['n_classes'] * torch.ones(
                    len(data['class_labels']), dtype=torch.long, device=self.device)
            
            pred = self.model.generate_samples(
                batch_size=gt_patch.shape[0],
                n_steps=self.model.zdm_n_steps,
                net_kwargs=net_kwargs,
                uncond_net_kwargs=uncond_net_kwargs,
                ema=ema
            )

            pred = (pred * 0.5 + 0.5).clamp(0, 1)
            gt_patch = (gt_patch * 0.5 + 0.5).clamp(0, 1)

            # FID
            for i in range(len(pred)):
                idx = int(os.environ['RANK']) + cnt * int(os.environ['WORLD_SIZE'])
                if max_samples is None or idx < max_samples: 
                    to_pil(pred[i]).save(os.path.join(cache_gen_dir, f'{idx}.png'))
                    to_pil(gt_patch[i]).save(os.path.join(cache_gt_dir, f'{idx}.png'))
                cnt += 1
        dist.barrier()

        if self.is_master:
            metrics = torch_fidelity.calculate_metrics(
                input1=cache_gen_dir,
                input2=cache_gt_dir,
                cuda=True,
                fid=True,
                verbose=False, 
            )
            prefix = 'eval_zdm' + ('_ema' if ema else '')
            ret = {
                f'{prefix}/FID': metrics['frechet_inception_distance'],
            }
        else:
            ret = {}
        dist.barrier()

        ret = {k: utils.Averager(v) for k, v in ret.items()}
        return ret
    
    def visualize_ae_fixset(self):
        if self.config.get('visualize_ae_dir') is None:
            return
        to_tensor = transforms.ToTensor()
        if self.is_master:
            files = sorted(os.listdir(self.config['visualize_ae_dir']))
            vis_images = []
            
            for f in files:
                image = Image.open(os.path.join(self.config['visualize_ae_dir'], f)).convert('RGB')
                x = to_tensor(image).unsqueeze(0).to(self.device)
                x = (x - 0.5) / 0.5
                gt_dummy = torch.zeros(x.shape[0], 7, x.shape[2], x.shape[3], device=self.device)
                
                pred1 = self.model({'inp': x, 'gt': gt_dummy}, mode='pred')
                pred2 = self.model({'inp': x, 'gt': gt_dummy}, mode='pred')
                vis_images.extend([x, pred1, pred2])
            
            vis_images = torch.cat(vis_images, dim=0)
            vis_images = torchvision.utils.make_grid(vis_images, normalize=True, value_range=(-1, 1), nrow=6)
            self.log_image('vis_ae_fixset', vis_images)
        dist.barrier()
    
    def visualize_ae_random(self):
        if self.is_master:
            idx_list = list(range(len(self.datasets['eval_ae'])))
            random.shuffle(idx_list)
            n_samples = self.config['visualize_ae_random_n_samples']
            vis_images = []
            
            for idx in idx_list[:n_samples]:
                data = self.datasets['eval_ae'][idx]
                for k, v in data.items():
                    data[k] = v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v
                
                pred1 = self.model(data, mode='pred')
                pred2 = self.model(data, mode='pred')
                gt_patch = data['gt'][:, :3, ...]
                vis_images.extend([gt_patch, pred1, pred2])
            
            vis_images = torch.cat(vis_images, dim=0)
            vis_images = torchvision.utils.make_grid(vis_images, normalize=True, value_range=(-1, 1), nrow=6)
            self.log_image('vis_ae_random', vis_images)
        dist.barrier()
    
    def visualize_zdm_fixset(self, ema):
        if self.is_master:
            vis_file = torch.load(self.config['visualize_zdm_file'], map_location='cpu')
            for k, v in vis_file.items():
                vis_file[k] = v.to(self.device) if torch.is_tensor(v) else v
            n_samples = len(vis_file['noise'])
            
            batch_size = self.config.get('visualize_zdm_batch_size', 1)
            guidance_list = [1.0] + self.config.get('visualize_zdm_guidance_list', [])

            vis_images = []
            
            for i in range(0, n_samples, batch_size):
                cur_batch_size = min(batch_size, n_samples - i)

                net_kwargs = dict()
                uncond_net_kwargs = dict()
                if self.config.get('visualize_zdm_setting') is not None:
                    setting = self.config['visualize_zdm_setting']
                    if setting['name'] == 'class':
                        net_kwargs['class_labels'] = vis_file['class_labels'][i:i + cur_batch_size]
                        uncond_net_kwargs['class_labels'] = setting['n_classes'] * torch.ones(
                            cur_batch_size, dtype=torch.long, device=self.device)
                    else:
                        raise NotImplementedError

                for guidance in guidance_list:
                    pred = self.model.generate_samples(
                        batch_size=cur_batch_size,
                        n_steps=self.model.zdm_n_steps,
                        net_kwargs=net_kwargs,
                        uncond_net_kwargs=uncond_net_kwargs,
                        ema=ema,
                        guidance=guidance,
                        noise=vis_file['noise'][i:i + cur_batch_size],
                    )
                    vis_images.append(pred)

            vis_images = torch.cat(vis_images, dim=0)
            vis_images = torchvision.utils.make_grid(vis_images, normalize=True, value_range=(-1, 1), nrow=batch_size)
            name = 'vis_zdm_fixset'
            name += '_ema' if ema else ''
            name += '_cfg' + str(guidance_list[1:])[1:-1] if len(guidance_list) > 1 else ''
            self.log_image(name, vis_images)
        dist.barrier()

    def visualize_zdm_random(self, ema):
        n_samples = self.config['visualize_zdm_random_n_samples']
        batch_size = self.config.get('visualize_zdm_batch_size', 1)
        guidance_list = [1.0] + self.config.get('visualize_zdm_guidance_list', [])

        vis_images = []

        if self.is_master:
            for i in range(0, n_samples, batch_size):
                cur_batch_size = min(batch_size, n_samples - i)

                net_kwargs = dict()
                uncond_net_kwargs = dict()
                if self.config.get('visualize_zdm_setting') is not None:
                    setting = self.config['visualize_zdm_setting']
                    if setting['name'] == 'class':
                        net_kwargs['class_labels'] = torch.randint(
                            setting['n_classes'], size=(cur_batch_size,), device=self.device)
                        uncond_net_kwargs['class_labels'] = setting['n_classes'] * torch.ones(
                            cur_batch_size, dtype=torch.long, device=self.device)
                    else:
                        raise NotImplementedError

                for guidance in guidance_list:
                    pred = self.model.generate_samples(
                        batch_size=cur_batch_size,
                        n_steps=self.model.zdm_n_steps,
                        net_kwargs=net_kwargs,
                        uncond_net_kwargs=uncond_net_kwargs,
                        ema=ema,
                        guidance=guidance,
                    )
                    vis_images.append(pred)

            vis_images = torch.cat(vis_images, dim=0)
            vis_images = torchvision.utils.make_grid(vis_images, normalize=True, value_range=(-1, 1), nrow=batch_size)
            name = 'vis_zdm_random'
            name += '_ema' if ema else ''
            name += '_cfg' + str(guidance_list[1:])[1:-1] if len(guidance_list) > 1 else ''
            self.log_image(name, vis_images)
        dist.barrier()

    def visualize_zdm_denoising(self, ema, n_selected_timesteps=5):
        if self.is_master:
            vis_file = torch.load(self.config['visualize_zdm_denoising_file'], map_location='cpu')

            vis_images = []
            
            for i in range(len(vis_file['inp'])):
                x = (
                    vis_file['inp'][i]
                    .to(self.device)
                    .unsqueeze(0)
                    .expand(n_selected_timesteps, -1, -1, -1)
                )
                
                z = self.model.encode(x)
                z = self.model.normalize_for_zdm(z)
                t = torch.linspace(0, 1, n_selected_timesteps + 1, device=self.device)[1:]
                noise = (
                    vis_file['noise'][i]
                    .to(self.device)
                    .unsqueeze(0)
                    .expand(n_selected_timesteps, -1, -1, -1)
                )
                z_t, _ = self.model.zdm_diffusion.add_noise(z, t, noise=noise)

                # Visualize noisy latents
                zp = self.model.denormalize_for_zdm(z_t)
                z_dec = self.model.decode(zp)
                coord, scale = make_coord_scale_grid(x.shape[-2:], device=self.device, batch_size=n_selected_timesteps)
                coord = coord.permute(0, 3, 1, 2)
                scale = scale.permute(0, 3, 1, 2)
                x_out = self.model.render(z_dec, coord, scale)
                vis_images.append(x_out)
                
                # Generate denoised latents
                net = self.model.zdm_net_ema if ema else self.model.zdm_net
                net_kwargs = dict()
                if self.config.get('visualize_zdm_setting') is not None:
                    setting = self.config['visualize_zdm_setting']
                    if setting['name'] == 'class':
                        net_kwargs['class_labels'] = (
                            vis_file['class_labels'][i]
                            .to(self.device)
                            .unsqueeze(0)
                            .expand(n_selected_timesteps)
                        )
                    else:
                        raise NotImplementedError
                pred = self.model.zdm_diffusion.get_prediction(net, z_t, t, net_kwargs=net_kwargs)
                zp = []
                for j in range(len(pred)):
                    zp.append(self.model.zdm_diffusion.convert_sample_prediction(z_t[j], float(t[j]), pred[j]))
                zp = torch.stack(zp, dim=0)

                # Visualize denoised latents
                zp = self.model.denormalize_for_zdm(zp)
                z_dec = self.model.decode(zp)
                coord, scale = make_coord_scale_grid(x.shape[-2:], device=self.device, batch_size=n_selected_timesteps)
                coord = coord.permute(0, 3, 1, 2)
                scale = scale.permute(0, 3, 1, 2)
                x_out = self.model.render(z_dec, coord, scale)
                vis_images.append(x_out)
            
            vis_images = torch.cat(vis_images, dim=0)
            vis_images = torchvision.utils.make_grid(vis_images, normalize=True, value_range=(-1, 1), nrow=n_selected_timesteps)
            self.log_image('vis_zdm' + ('_ema' if ema else '') + '_denoising', vis_images)
        dist.barrier() 
