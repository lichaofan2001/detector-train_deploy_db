"""
Testing CLI for detector package.

This module provides the command line interface for testing/evaluating object detection models.
"""

import sys
import argparse
from pathlib import Path
import os
import numpy as np

from detector.test import test, parse_opt, get_parser
from detector.utils.general import check_file
from detector.utils.plots import plot_study_txt

# Import path utilities for resolving default paths
from detector import (
    get_default_weights_path,
    get_default_data_path,
)


def resolve_default_paths(opt):
    """
    Resolve default paths for weights and data if not specified.
    
    Args:
        opt: Options namespace from argparse
    
    Returns:
        opt: Options namespace with resolved paths
    """
    # Resolve default weights path
    if opt.weights is None:
        opt.weights = [str(get_default_weights_path())]
        print(f"Using default weights: {opt.weights}")
    elif isinstance(opt.weights, str):
        opt.weights = [opt.weights]
    
    # Resolve default data path
    if opt.data is None:
        opt.data = str(get_default_data_path())
        print(f"Using default data: {opt.data}")
    
    return opt


def test_command():
    """
    Command line entry point for testing.
    
    This function is called when the user runs 'detector-test' command.
    It parses command line arguments and starts the testing process.
    """
    parser = get_parser()
    opt = parser.parse_args()
    
    # Resolve default paths before any other processing
    opt = resolve_default_paths(opt)
    
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    
    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             matrix_conf_thresh=opt.matrix_conf_thresh, opt=opt)

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)


if __name__ == '__main__':
    test_command()
