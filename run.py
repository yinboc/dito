import argparse
import os

from omegaconf import OmegaConf

from trainers import trainers_dict


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/_.yaml')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', '-t', default=None)
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--force-replace', '-f', action='store_true')
    parser.add_argument('--wandb', '-w', action='store_true')
    parser.add_argument('--save-root', default='save')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()
    return args


def parse_config(config):
    if config.get('__base__') is not None:
        filenames = config.pop('__base__')
        if isinstance(filenames, str):
            filenames = [filenames]
        base_config = OmegaConf.merge(*[
            parse_config(OmegaConf.load(_))
            for _ in filenames
        ])
        config = OmegaConf.merge(base_config, config)
    return config


def make_env(args):
    env = dict()
    
    if args.name is None:
        exp_name = os.path.splitext(os.path.basename(args.config))[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag
    env['exp_name'] = exp_name
    
    env['save_dir'] = os.path.join(args.save_root, exp_name)
    env['wandb'] = args.wandb
    env['resume'] = args.resume
    env['force_replace'] = args.force_replace
    return env


if __name__ == '__main__':
    args = make_args()
    env = make_env(args)
    config = parse_config(OmegaConf.load(args.config))
    trainer = trainers_dict[config.trainer](env, config)
    trainer.run(eval_only=args.eval_only)
