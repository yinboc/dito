import os
import time
import copy
from datetime import timedelta

import yaml
import wandb
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import datasets
import models
import utils
from .trainers import register


@register('base_trainer')
class BaseTrainer():

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.config_dict = OmegaConf.to_container(config, resolve=True)

        if config.get('allow_tf32', False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=240))
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.node_id = int(os.environ['GROUP_RANK'])
        self.node_tot = self.world_size // int(os.environ['LOCAL_WORLD_SIZE'])
        self.is_master = (self.rank == 0)
        
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', torch.cuda.current_device())

        if self.is_master:
            # Setup path
            if env['resume']:
                replace = False
                force_replace = False
            else:
                replace = True
                force_replace = env['force_replace']
            utils.ensure_path(env['save_dir'], replace=replace, force_replace=force_replace)
            
            # Save config
            with open(os.path.join(env['save_dir'], 'config.yaml'), 'w') as f:
                yaml.dump(self.config_dict, f, sort_keys=False)

            # Setup logging
            logger = utils.set_logger(os.path.join(env['save_dir'], 'log.txt'))
            self.log = logger.info
        
            # Setup wandb
            if env['wandb']:
                self.enable_wandb = True
                os.environ['WANDB_NAME'] = env['exp_name']
                os.environ['WANDB_DIR'] = env['save_dir']
                with open('load/wandb.yaml', 'r') as f:
                    wandb_config = yaml.load(f, Loader=yaml.FullLoader)
                os.environ['WANDB_API_KEY'] = wandb_config['api_key']
                wandb.init(project=wandb_config['project'], entity=wandb_config['entity'], config=self.config_dict, resume=True)
            else:
                self.enable_wandb = False
        else:
            self.log = lambda *args, **kwargs: None
            self.enable_wandb = False
        dist.barrier()

        self.log(f'Environment setup done. World size: {self.world_size}.')

    def run(self, eval_only=False):
        self.make_datasets()

        resume_ckpt = os.path.join(self.env['save_dir'], 'ckpt-last.pth')
        resume = (self.env['resume'] and os.path.isfile(resume_ckpt))
        if resume:
            self.resume_ckpt = torch.load(resume_ckpt, map_location='cpu')
        else:
            self.resume_ckpt = None

        self.make_model()
        if resume:
            self.model.load_state_dict(self.resume_ckpt['model']['sd'])
            self.resume_ckpt['model'] = None
            self.log(f'Resumed model from checkpoint {resume_ckpt}.')

        if eval_only:
            self.model_ddp = self.model
            with torch.no_grad():
                self.log_buffer = [f'Eval']
                self.iter = 0
                self.evaluate()
                self.visualize()
                self.log(', '.join(self.log_buffer))
        
        else:
            self.model_ddp = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.config.get('find_unused_parameters', False)
            )
            
            self.make_optimizers()
            if resume:
                for name, optimizer in self.resume_ckpt['optimizers'].items():
                    self.optimizers[name].load_state_dict(optimizer['sd'])
                self.resume_ckpt['optimizers'] = None
                self.log(f'Resumed optimizers.')

            self.run_training()

        if self.enable_wandb:
            wandb.finish()

    def make_distributed_loader(self, dataset, batch_size, shuffle, drop_last, num_workers, pin_memory):
        assert batch_size % self.world_size == 0
        assert num_workers % self.world_size == 0
        if isinstance(dataset, IterableDataset):
            sampler = None
        else:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        loader = DataLoader(
            dataset,
            batch_size=batch_size // self.world_size,
            drop_last=drop_last,
            sampler=sampler,
            num_workers=num_workers // self.world_size,
            pin_memory=pin_memory
        )
        return loader, sampler

    def make_datasets(self):
        self.datasets = dict()
        self.loaders = dict()
        self.loader_samplers = dict()

        for split, spec in self.config.datasets.items():
            loader_spec = spec.pop('loader')
            
            dataset = datasets.make(spec)
            self.datasets[split] = dataset
            if isinstance(dataset, IterableDataset):
                self.log(f'Dataset {split}: IterableDataset')
            else:
                self.log(f'Dataset {split}: len={len(dataset)}')

            drop_last = loader_spec.get('drop_last', True)
            shuffle = loader_spec.get('shuffle', True)
            self.loaders[split], self.loader_samplers[split] = self.make_distributed_loader(
                dataset,
                loader_spec.batch_size,
                shuffle,
                drop_last,
                loader_spec.num_workers,
                loader_spec.get('pin_memory', True)
            )
    
    def make_model(self):
        model = models.make(self.config.model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = model.to(self.device)
        self.log(f'Model: #params={utils.compute_num_params(model)}')

    def make_optimizers(self):
        self.optimizers = {'model': utils.make_optimizer(self.model.parameters(), self.config.optimizers['model'])}

    def run_training(self):
        config = self.config
        max_iter = config['max_iter']
        epoch_iter = config['epoch_iter']
        assert max_iter % epoch_iter == 0
        max_epoch = max_iter // epoch_iter

        save_iter = config.get('save_iter')
        if save_iter is not None:
            assert save_iter % epoch_iter == 0
            save_epoch = save_iter // epoch_iter
        else:
            save_epoch = max_epoch + 1

        eval_iter = config.get('eval_iter')
        if eval_iter is not None:
            assert eval_iter % epoch_iter == 0
            eval_epoch = eval_iter // epoch_iter
        else:
            eval_epoch = max_epoch + 1

        vis_iter = config.get('vis_iter')
        if vis_iter is not None:
            assert vis_iter % epoch_iter == 0
            vis_epoch = vis_iter // epoch_iter
        else:
            vis_epoch = max_epoch + 1

        if config.get('ckpt_select_metric') is not None:
            m = config.ckpt_select_metric
            self.ckpt_select_metric = m.name
            self.ckpt_select_type = m.type
            if m.type == 'min':
                self.ckpt_select_v = 1e18
            elif m.type == 'max':
                self.ckpt_select_v = -1e18
        else:
            self.ckpt_select_metric = None
            self.ckpt_select_v = 0

        self.train_loader = self.loaders['train']
        self.train_loader_sampler = self.loader_samplers['train']
        self.train_loader_epoch = 0
        self.train_loader_iter = None

        self.iter = 0

        if self.resume_ckpt is not None:
            for _ in range(self.resume_ckpt['iter']):
                self.iter += 1
                self.at_train_iter_start()
            self.ckpt_select_v = self.resume_ckpt['ckpt_select_v']
            self.train_loader_epoch = self.resume_ckpt['train_loader_epoch']
            self.train_loader_iter = None
            self.resume_ckpt = None
            self.log(f'Resumed iter status.')
        
        if config.get('vis_before_training', False):
            self.visualize()

        start_epoch = self.iter // epoch_iter + 1
        epoch_timer = utils.EpochTimer(max_epoch - start_epoch + 1)

        for epoch in range(start_epoch, max_epoch + 1):
            self.log_buffer = [f'Epoch {epoch}']

            for sampler in self.loader_samplers.values():
                if sampler is not self.train_loader_sampler:
                    sampler.set_epoch(epoch)

            self.model_ddp.train()

            ave_scalars = dict()
            pbar = range(1, epoch_iter + 1)
            if self.is_master and epoch == start_epoch:
                pbar = tqdm(pbar, desc='train', leave=False)

            t_data = 0
            t_nondata = 0
            t_before_data = time.time()
            
            for _ in pbar:
                self.iter += 1
                self.at_train_iter_start()

                try:
                    if self.train_loader_iter is None:
                        raise StopIteration
                    data = next(self.train_loader_iter)
                except StopIteration:
                    self.train_loader_epoch += 1
                    self.train_loader_sampler.set_epoch(self.train_loader_epoch)
                    self.train_loader_iter = iter(self.train_loader)
                    data = next(self.train_loader_iter)
                
                t_after_data = time.time()
                t_data += t_after_data - t_before_data
                
                for k, v in data.items():
                    data[k] = v.to(self.device) if torch.is_tensor(v) else v
                
                ret = self.train_step(data)

                bs = len(next(iter(data.values())))
                for k, v in ret.items():
                    if ave_scalars.get(k) is None:
                        ave_scalars[k] = utils.Averager()
                    ave_scalars[k].add(v, n=bs)
                
                t_before_data = time.time()
                t_nondata += t_before_data - t_after_data

                if self.is_master and epoch == start_epoch:
                    pbar.set_description(desc=f'train: loss={ret["loss"]:.4f}')
            
            self.save_ckpt('ckpt-last.pth')

            self.sync_ave_scalars(ave_scalars)

            logtext = 'train:'
            for k, v in ave_scalars.items():
                logtext += f' {k}={v.item():.4f}'
                self.log_scalar('train/' + k, v.item())
            logtext += f' (d={t_data / (t_data + t_nondata):.2f})'
            self.log_buffer.append(logtext)

            if epoch % save_epoch == 0 and epoch != max_epoch:
                self.save_ckpt(f'ckpt-{self.iter}.pth')

            if epoch % eval_epoch == 0:
                with torch.no_grad():
                    eval_ave_scalars = self.evaluate()
                if self.ckpt_select_metric is not None:
                    v = eval_ave_scalars[self.ckpt_select_metric].item()
                    if ((self.ckpt_select_type == 'min' and v < self.ckpt_select_v) or
                        (self.ckpt_select_type == 'max' and v > self.ckpt_select_v)):
                        self.ckpt_select_v = v
                        self.save_ckpt('ckpt-best.pth')

            if epoch % vis_epoch == 0:
                with torch.no_grad():
                    self.visualize()

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            self.log_buffer.append(f'{epoch_time} {tot_time}/{est_time}')
            
            self.log(', '.join(self.log_buffer))

    def at_train_iter_start(self):
        pass

    def train_step(self, data, bp=True):
        if self.config.get('autocast_bfloat16', False):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                ret = self.model_ddp(data)
        else:
            ret = self.model_ddp(data)
        
        loss = ret.pop('loss')
        ret['loss'] = loss.item()
        if bp:
            self.model_ddp.zero_grad()
            loss.backward()
            for o in self.optimizers.values():
                o.step()
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

        logtext = 'val:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_scalar('val/' + k, v.item())
        self.log_buffer.append(logtext)

        return ave_scalars

    def visualize(self):
        pass

    def save_ckpt(self, filename):
        if self.is_master:
            model_spec = copy.copy(self.config_dict['model'])
            model_spec['sd'] = self.model.state_dict()
            optimizers_spec = dict()
            for name, spec in self.config_dict['optimizers'].items():
                spec = copy.copy(spec)
                spec['sd'] = self.optimizers[name].state_dict()
                optimizers_spec[name] = spec
            ckpt = {
                'config': self.config_dict,
                'model': model_spec,
                'optimizers': optimizers_spec,
                'iter': self.iter,
                'train_loader_epoch': self.train_loader_epoch,
                'ckpt_select_v': self.ckpt_select_v,
            }
            torch.save(ckpt, os.path.join(self.env['save_dir'], filename))
        dist.barrier()

    def sync_ave_scalars(self, ave_scalars):
        keys = sorted(list(ave_scalars.keys()))
        for k in keys:
            if not k.startswith('_'):
                v = ave_scalars[k]
                vt = torch.tensor(v.item(), device=self.device)
                dist.all_reduce(vt, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                ave_scalars[k].v = vt.item() / self.world_size
                ave_scalars[k].n *= self.world_size

    def log_scalar(self, k, v):
        if self.enable_wandb:
            wandb.log({k: v}, step=self.iter)

    def log_image(self, k, v):
        if self.enable_wandb:
            wandb.log({k: wandb.Image(v)}, step=self.iter)
