import os
import shutil
import time
import logging

from torch.optim import SGD, Adam, AdamW


def ensure_path(path, replace=True, force_replace=False):
    is_temp = os.path.basename(path.rstrip('/')).startswith('_')
    if os.path.exists(path):
        if replace and (is_temp or force_replace or input(f'{path} exists, replace? y/[n] ') == 'y'):
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.makedirs(path)


def set_logger(file_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, 'a')
    formatter = logging.Formatter('[%(asctime)s] %(message)s', '%m-%d %H:%M:%S')
    for handler in [stream_handler, file_handler]:
        handler.setFormatter(formatter)
        handler.setLevel('INFO')
        logger.addHandler(handler)
    return logger


def compute_num_params(model, text=True):
    tot = sum(p.numel() for p in model.parameters())
    if text:
        if tot >= 1e6:
            s = '{:.1f}M'.format(tot / 1e6)
        else:
            s = '{:.1f}K'.format(tot / 1e3)
        return f'{s} ({tot})'
    else:
        return tot


def make_optimizer(params, optimizer_spec):
    optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
    }[optimizer_spec['name']](params, **optimizer_spec['args'])
    return optimizer


class Averager():

    def __init__(self, v=None):
        if v is None:
            self.n = 0.
            self.v = 0.
        else:
            self.n = 1.
            self.v = v

    def add(self, v, n=1.0):
        self.v = self.v * (self.n / (self.n + n)) + v * (n / (self.n + n))
        self.n += n

    def item(self):
        return self.v


class EpochTimer():

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.epoch = 0
        self.t_start = time.time()
        self.t_last = self.t_start

    def epoch_done(self):
        t_cur = time.time()
        self.epoch += 1
        epoch_time = t_cur - self.t_last
        tot_time = t_cur - self.t_start
        est_time = tot_time / self.epoch * self.max_epoch
        self.t_last = t_cur
        return time_text(epoch_time), time_text(tot_time), time_text(est_time)


def time_text(sec):
    if sec >= 3600:
        return f'{sec / 3600:.1f}h'
    elif sec >= 60:
        return f'{sec / 60:.1f}m'
    else:
        return f'{sec:.1f}s'
