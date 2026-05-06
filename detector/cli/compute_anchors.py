"""
Compute anchors CLI for detector package.

This module provides the command line interface for computing optimal anchors
for a dataset using k-means clustering.
"""

import argparse

from detector.utils import autoanchor


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='使用 k-means 聚类计算数据集的最优锚框')
    parser.add_argument('--data', type=str, required=True, help='数据集配置 YAML 文件路径（例如：data/dataset.yaml）')
    parser.add_argument('--num-anchors', type=int, default=9, help='要计算的锚框数量（默认：9）')
    parser.add_argument('--img-size', type=int, default=1024, help='锚框计算的图像尺寸（默认：1024）')
    parser.add_argument('--thresh', type=float, default=5.0, help='k-means 的 IoU 阈值（默认：5.0）')
    parser.add_argument('--iters', type=int, default=5000, help='k-means 迭代次数（默认：5000）')
    parser.add_argument('--verbose', action='store_true', help='打印详细输出')
    return parser.parse_args()


def compute_anchors(data, num_anchors=9, img_size=1024, thresh=5.0, iters=5000, verbose=False):
    """
    Compute optimal anchors for a dataset using k-means clustering.
    
    Args:
        data: Path to data config YAML file
        num_anchors: Number of anchors to compute (default: 9)
        img_size: Image size for anchor computation (default: 1024)
        thresh: IoU threshold for k-means (default: 5.0)
        iters: Number of k-means iterations (default: 5000)
        verbose: Print verbose output (default: False)
    
    Returns:
        Computed anchors as a numpy array
    """
    print(f'Computing {num_anchors} anchors for dataset: {data}')
    print(f'Image size: {img_size}, IoU threshold: {thresh}, Iterations: {iters}')
    
    new_anchors = autoanchor.kmean_anchors(data, num_anchors, img_size, thresh, iters, verbose)
    
    print('Computed anchors:')
    print(new_anchors)
    
    return new_anchors


def compute_anchors_command():
    """
    Command line entry point for computing anchors.
    
    This function is called when the user runs 'compute-anchors' command.
    It parses command line arguments and computes optimal anchors for the dataset.
    """
    opt = parse_opt()
    
    compute_anchors(
        data=opt.data,
        num_anchors=opt.num_anchors,
        img_size=opt.img_size,
        thresh=opt.thresh,
        iters=opt.iters,
        verbose=opt.verbose
    )
