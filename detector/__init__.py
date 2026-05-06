"""
Detector - Object Detection Model Training and Inference Package

This package provides tools for training and evaluating object detection models,
based on YOLOv7 architecture.
"""

import os
from pathlib import Path

__version__ = '0.1.0'
__author__ = 'Detector Team'


def get_package_root():
    """
    Get the root directory of the detector package.
    
    Returns:
        Path: Absolute path to the detector package root directory.
    """
    return Path(__file__).parent


def get_cfg_path(filename=None):
    """
    Get the absolute path to a configuration file in the cfg directory.
    
    Args:
        filename (str, optional): Name of the configuration file. If None, returns the cfg directory path.
    
    Returns:
        Path: Absolute path to the configuration file or directory.
    """
    cfg_dir = get_package_root() / 'cfg'
    if filename:
        return cfg_dir / filename
    return cfg_dir


def get_data_path(filename=None):
    """
    Get the absolute path to a data file in the data directory.
    
    Args:
        filename (str, optional): Name of the data file. If None, returns the data directory path.
    
    Returns:
        Path: Absolute path to the data file or directory.
    """
    data_dir = get_package_root() / 'data'
    if filename:
        return data_dir / filename
    return data_dir


def get_hyp_path(filename=None):
    """
    Get the absolute path to a hyperparameters file in the data directory.
    
    Args:
        filename (str, optional): Name of the hyperparameters file. If None, returns the hyp directory path.
    
    Returns:
        Path: Absolute path to the hyperparameters file or directory.
    """
    hyp_dir = get_package_root() / 'data'
    if filename:
        return hyp_dir / filename
    return hyp_dir


def get_detector_home():
    """
    Get the detector home directory for storing user-specific files.
    
    Returns:
        Path: Absolute path to the detector home directory (~/.detector).
    """
    home_dir = Path.home() / '.detector'
    home_dir.mkdir(parents=True, exist_ok=True)
    return home_dir


def get_ckpoints_path(filename=None):
    """
    Get the absolute path to a checkpoint file in the user's detector ckpoints directory.
    
    Args:
        filename (str, optional): Name of the checkpoint file. If None, returns the ckpoints directory path.
    
    Returns:
        Path: Absolute path to the checkpoint file or directory.
    """
    ckpoints_dir = get_detector_home() / 'ckpoints'
    ckpoints_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        return ckpoints_dir / filename
    return ckpoints_dir


# Default paths for training
DEFAULT_CFG = 'yolov7-tiny-silu-sqnet.yaml'
DEFAULT_WEIGHTS = 'opencar.pt'
DEFAULT_DATA = 'data_test.yaml'
DEFAULT_HYP_SCRATCH = 'hyp.scratch.tiny.yaml'
DEFAULT_HYP_FINETUNE = 'hyp.finetune.tiny.yaml'


def get_default_cfg_path():
    """
    Get the default configuration file path.
    
    Returns:
        Path: Absolute path to the default configuration file.
    """
    return get_cfg_path(DEFAULT_CFG)


def get_default_weights_path():
    """
    Get the default weights file path.
    
    Returns:
        Path: Absolute path to the default weights file.
    """
    return get_ckpoints_path(DEFAULT_WEIGHTS)


def get_default_data_path():
    """
    Get the default data file path.
    
    Returns:
        Path: Absolute path to the default data file.
    """
    return get_data_path(DEFAULT_DATA)


def get_default_hyp_path(scratch=True):
    """
    Get the default hyperparameters file path.
    
    Args:
        scratch (bool): If True, return scratch hyp path; otherwise return finetune hyp path.
    
    Returns:
        Path: Absolute path to the default hyperparameters file.
    """
    filename = DEFAULT_HYP_SCRATCH if scratch else DEFAULT_HYP_FINETUNE
    return get_hyp_path(filename)


# Initialize module loader with default mappings for backward compatibility
# This ensures old .pt model files (saved with paths like 'models.yolo.Model')
# can be loaded after the models folder was moved into the detector package
from detector.utils.module_loader import register_module_mapping

# Register default prefix mappings for backward compatibility
register_module_mapping('models', 'detector.models')
register_module_mapping('utils', 'detector.utils')
register_module_mapping('cli', 'detector.cli')

# Import core functions for easy access
from detector.train import train
from detector.test import test
from detector.show_detector_results import show_command
from detector.predict_bbox import predict_command

__all__ = [
    'train', 
    'test', 
    'show_command', 
    'predict_command',
    'get_package_root',
    'get_cfg_path',
    'get_data_path',
    'get_hyp_path',
    'get_detector_home',
    'get_ckpoints_path',
    'get_default_cfg_path',
    'get_default_weights_path',
    'get_default_data_path',
    'get_default_hyp_path',
    'DEFAULT_CFG',
    'DEFAULT_WEIGHTS',
    'DEFAULT_DATA',
    'DEFAULT_HYP_SCRATCH',
    'DEFAULT_HYP_FINETUNE',
]
