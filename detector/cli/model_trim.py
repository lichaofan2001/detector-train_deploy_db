"""
Model trim CLI for detector package.

This module provides the command line interface for trimming classes from a trained model.
"""

import torch
import os
import argparse
from detector.utils.general import strip_optimizer
from detector.utils.tools import check_path_exist
from detector.utils.module_loader import torch_load


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='裁剪目标检测模型的类别')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件路径 (.pt 文件)')
    parser.add_argument('--to-trim', type=str, required=True, help='要裁剪的类别索引，逗号分隔（例如："3,4"）')
    parser.add_argument('--out-file', type=str, default=None, help='输出文件路径（默认：权重目录带 trim 后缀）')
    parser.add_argument('--force', action='store_true', help='无需确认直接覆盖输出文件')
    return parser.parse_args()


def trim_model(weights, to_trim, out_file=None, force=False):
    """
    Trim specified classes from the model and save the trimmed weights.
    
    Args:
        weights: Path to the model weights file (.pt)
        to_trim: List of class indices to trim
        out_file: Path to save the trimmed model (optional)
        force: Overwrite output file without prompting (optional)
    
    Returns:
        Path to the saved trimmed model
    """

    if not os.path.exists(weights):
        raise FileNotFoundError(f'Weights file not found: {weights}')

    if not os.path.isfile(weights):
        raise ValueError(f'Weights must be a file, got: {weights}')

    if not weights.endswith('.pt'):
        raise ValueError(f'Weights file must be a .pt file, got: {weights}')

    if not isinstance(to_trim, list):
        raise ValueError(f'to_trim must be a list, got: {type(to_trim)}')

    if not all(isinstance(i, int) for i in to_trim):
        raise ValueError(f'all elements in to_trim must be integers, got: {to_trim}')
    if out_file is not None and not out_file.endswith('.pt'):
        raise ValueError(f'Output file must be a .pt file, got: {out_file}')
    if out_file is not None and not os.path.isabs(out_file):
        raise ValueError(f'Output file must be an absolute path, got: {out_file}')

    # Load model to get nc (number of classes) with backward-compatible module path resolution
    state_dict = torch_load(weights, map_location='cpu')
    model = state_dict['model']
    yaml = model.yaml
    nc = yaml['nc']

    # Filter valid class indices to trim
    to_trim_new = []
    for i in to_trim:
        if i >= 0 and i < nc:
            to_trim_new.append(i)
        else:
            print('invalid class index: {}, nc={}'.format(i, nc))

    # Determine output file path
    if out_file is None:
        pn, _ = os.path.split(weights)
        to_trim_str = '_'.join([str(i) for i in to_trim_new])
        out_file = os.path.join(pn, f'best_trim_{to_trim_str}.pt')

    # Check if output file exists and handle overwrite
    if os.path.exists(out_file):
        if not force:
            print(f'Output file {out_file} already exists. Use --force to overwrite.')
            exit(1)

    # Ensure output directory exists
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    out_pn = os.path.split(out_file)[0]
    check_path_exist(out_pn)

    # Strip optimizer and load stripped weights
    tmp_file = os.path.join(out_pn, 'tmp.pt')
    try:
        strip_optimizer(weights, tmp_file)  # Strip optimizer info
        state_dict = torch_load(tmp_file, map_location='cpu')  # Load stripped weights with module path remapping
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    model = state_dict['model']
    yaml = model.yaml

    names = model.names
    names = [names[i] for i in range(len(names)) if i not in to_trim_new]
    model.names = names

    new_nc = nc - len(to_trim_new)
    yaml['nc'] = new_nc

    model.nc = new_nc

    det_layer = model.model[-1]
    det_layer.nc = new_nc
    det_layer.no = new_nc + 5

    # Update convolution weights
    nl = det_layer.nl  # number of detection layers
    na = det_layer.na  # number of anchors

    remaining_axis = []
    # Add in order
    for j in range(na):
        offset = j * (nc + 5)
        remaining_axis.append(0 + offset)
        remaining_axis.append(1 + offset)
        remaining_axis.append(2 + offset)
        remaining_axis.append(3 + offset)
        remaining_axis.append(4 + offset)

        for i in range(nc):
            if i not in to_trim_new:
                remaining_axis.append(5 + i + offset)

    print(remaining_axis)
    for i in range(nl):
        det_layer.m[i].weight = torch.nn.Parameter(det_layer.m[i].weight[remaining_axis, :, :, :])
        det_layer.m[i].bias = torch.nn.Parameter(det_layer.m[i].bias[remaining_axis])

    # Print shapes
    for i in range(nl):
        print(state_dict['model'].model[-1].m[i].weight.shape)


    torch.save(state_dict, out_file)
    print(f'Trimmed model saved to: {out_file}')
    return out_file


def model_trim_command():
    """
    Command line entry point for trimming model classes.
    
    This function is called when the user runs 'detector-trim' command.
    It parses command line arguments and trims the specified classes from the model.
    """
    opt = parse_opt()

    # Parse to_trim from comma-separated string to list of integers
    to_trim = [int(x.strip()) for x in opt.to_trim.split(',')]

    trim_model(opt.weights, to_trim, opt.out_file, opt.force)
