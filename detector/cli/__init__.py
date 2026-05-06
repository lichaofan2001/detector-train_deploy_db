"""
Command Line Interface module for detector package.

This module provides CLI entry points for training and testing object detection models.
"""

from detector.cli.train_cli import train_command
from detector.cli.test_cli import test_command
from detector.cli.detect import detect_command
from detector.cli.compute_anchors import compute_anchors_command
from detector.cli.print_anchors import print_anchors_command
from detector.cli.model_trim import model_trim_command
from detector.cli.model_rename_classes import model_rename_command
try:
    from detector.cli.export_onnx import export_onnx_command
    from detector.cli.export_fake_rgbt import export_fake_rgbt_command
    from detector.cli.export_scale_conf import export_scale_conf_command
except:
    print('onnx failed')
from detector.cli.analyze_list import analyze_list_command
from detector.cli.generate_templates import generate_templates_command

__all__ = ['train_command', 'test_command', 'detect_command', 'compute_anchors_command', 'print_anchors_command', 'model_trim_command', 'model_rename_command', 'export_onnx_command', 'export_fake_rgbt_command', 'export_scale_conf_command', 'analyze_list_command', 'generate_templates_command']
