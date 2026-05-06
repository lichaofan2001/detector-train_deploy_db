"""
Model rename classes CLI for detector package.

This module provides the command line interface for renaming classes in a trained model.
"""

import torch
import os
import argparse
from detector.utils.general import strip_optimizer
from detector.utils.module_loader import torch_load


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='重命名目标检测模型中的类别')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件路径 (.pt 文件)')
    parser.add_argument('--newnames', type=str, required=True, help='新的类别名称，逗号分隔（例如："car,person,dog"）')
    parser.add_argument('--out-file', type=str, default=None, help='输出文件路径（默认：权重目录中的 model_newname.pt）')
    return parser.parse_args()


def rename_classes(weights, newnames, out_file=None):
    """
    Rename classes in the model and save the updated weights.
    
    Args:
        weights: Path to the model weights file (.pt)
        newnames: List of new class names
        out_file: Path to save the renamed model (optional)
    
    Returns:
        Path to the saved renamed model
    """
    # Strip optimizer in place before loading
    strip_optimizer(weights)

    # Load stripped weights with backward-compatible module path resolution
    state_dict = torch_load(weights, map_location='cpu')

    model = state_dict['model']
    if 'ema' in state_dict:
        state_dict.pop('ema')

    yaml = model.yaml

    nc = yaml['nc']

    assert nc == len(newnames), f"Number of classes ({nc}) does not match number of new names ({len(newnames)})"

    model.names = newnames

    if out_file is None:
        pn = os.path.dirname(weights)
        out_file = os.path.join(pn, f'model_newname.pt')

    torch.save(state_dict, out_file)
    print(f'Renamed model saved to: {out_file}')
    return out_file


def model_rename_command():
    """
    Command line entry point for renaming model classes.
    
    This function is called when the user runs 'detector-rename' command.
    It parses command line arguments and renames the classes in the model.
    """
    opt = parse_opt()

    # Parse newnames from comma-separated string to list of strings
    newnames = [x.strip() for x in opt.newnames.split(',')]

    rename_classes(opt.weights, newnames, opt.out_file)
