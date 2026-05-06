"""
Print anchors CLI for detector package.

This module provides the command line interface for printing anchors from a trained model.
"""

import torch
import argparse
from detector.utils.module_loader import torch_load


def read_anchors(weights):
    """
    Read anchors from a trained model weights file.
    
    Args:
        weights: Path to the model weights file (.pt)
    
    Returns:
        anchors: Tensor containing the anchor boxes
    """
    state_dict = torch_load(weights, map_location='cpu')

    model = state_dict['model']

    anchors = model.model[-1].anchors
    strides = model.model[-1].stride.reshape((-1, 1, 1))
    anchors = anchors * strides
    
    return anchors


def print_anchors(anchors):
    """
    Print anchors in a formatted way.
    
    Args:
        anchors: Tensor containing the anchor boxes
    """
    for i, anchor in enumerate(anchors):
        for point in anchor:
            print(f'{point[0]:.2f},{point[1]:.2f}', end='  ')
        print()


def print_anchors_command():
    """
    Command line entry point for printing anchors.
    
    This function is called when the user runs 'print-anchors' command.
    It parses command line arguments and prints the anchors from the model.
    """
    parser = argparse.ArgumentParser(description='打印模型中的锚框')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件路径 (.pt 文件)')
    opt = parser.parse_args()

    weights = opt.weights
    
    anchors = read_anchors(weights)
    print_anchors(anchors)
