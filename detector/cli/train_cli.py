"""
Training CLI for detector package.

This module provides the command line interface for training object detection models.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

from detector.train import train, parse_opt, get_parser
from detector.utils.general import (
    check_file, increment_path, set_logging, check_git_status,
    check_requirements, colorstr
)
from detector.utils.torch_utils import select_device
from detector.utils.wandb_logging.wandb_utils import check_wandb_resume
import yaml
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import logging

# Import path utilities for resolving default paths
from detector import (
    get_default_cfg_path,
    get_default_weights_path,
    get_default_data_path,
    get_default_hyp_path,
)

logger = logging.getLogger(__name__)


def resolve_default_paths(opt):
    """
    Resolve default paths for cfg, weights, data, and hyp if not specified.
    
    Args:
        opt: Options namespace from argparse
    
    Returns:
        opt: Options namespace with resolved paths
    """
    # Resolve default cfg path
    if opt.cfg is None:
        opt.cfg = str(get_default_cfg_path())
        logger.info(f"Using default cfg: {opt.cfg}")
    
    # Resolve default weights path
    if opt.weights is None:
        opt.weights = str(get_default_weights_path())
        logger.info(f"Using default weights: {opt.weights}")
    
    # Resolve default data path
    if opt.data is None:
        opt.data = str(get_default_data_path())
        logger.info(f"Using default data: {opt.data}")
    
    # Resolve default hyp path (prefer scratch hyp by default)
    if opt.hyp is None:
        opt.hyp = str(get_default_hyp_path(scratch=True))
        logger.info(f"Using default hyp: {opt.hyp}")
    
    return opt


def train_command():
    """
    Command line entry point for training.
    
    This function is called when the user runs 'detector-train' command.
    It parses command line arguments and starts the training process.
    """
    parser = get_parser()
    opt = parser.parse_args()
    
    # Resolve default paths before any other processing
    opt = resolve_default_paths(opt)
    
    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    
    set_logging(opt.global_rank)
    
    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        from ..utils.general import get_latest_run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (1, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (1, 2.0, 10.0),  # anchors per output grid (3 to 10)
                'fl_gamma': (1, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (1, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (1, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mosaic (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        bounds = {k: v[1:] for k, v in meta.items()}  # (lower, upper) tuple
        n = len(meta)  # number of hyperparameters
        x = np.random.uniform(bounds[k][0], bounds[k][1], n)  # random values
        print(f'Hyperparameter evolution starting with {n} parameters')
        print(f'Initial values: {dict(zip(meta.keys(), x))}')
        # Note: Full evolution logic would continue here
        # For now, we just print a message
        print('Hyperparameter evolution is a work in progress. Please use manual training for now.')


if __name__ == '__main__':
    train_command()
