"""
Export Fake RGBT ONNX CLI for detector package.

This module provides the command line interface for exporting PyTorch models to ONNX format
with fake RGBT input (RGB + IR channels).
"""

import sys
import time
import torch
import torch.nn as nn

from detector.models.experimental import attempt_load
from detector.utils.general import set_logging, check_img_size
from detector.utils.torch_utils import select_device
from detector.utils.activations import Hardswish, SiLU


class FakeRgbt(nn.Module):
    """Fake RGBT model that takes RGB and IR images as input."""
    
    def __init__(self, model):
        super().__init__()
        self.rgb = model
        
    def forward(self, img, img_ir):
        y = self.rgb(img + img_ir * 0.0)
        return y


def parse_opt():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='导出 PyTorch 模型到 Fake RGBT ONNX 格式')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='图像尺寸（高，宽）')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--dynamic', action='store_true', help='动态 ONNX 轴')
    parser.add_argument('--dynamic-batch', action='store_true', help='动态批次大小（用于 TensorRT 和 ONNX Runtime）')
    parser.add_argument('--grid', action='store_true', help='导出 Detect() 层网格')
    parser.add_argument('--device', default='cpu', help='CUDA 设备，如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--simplify', action='store_true', help='简化 ONNX 模型')
    parser.add_argument('--include-nms', action='store_true', help='导出带 NMS 的端到端 ONNX')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset 版本')
    return parser.parse_args()


def export_fake_rgbt(weights, img_size, batch_size, dynamic=False, dynamic_batch=False,
                     grid=False, device='cpu', simplify=False, include_nms=False, opset=12):
    """
    Export PyTorch model to Fake RGBT ONNX format.
    
    Args:
        weights: Path to the model weights file (.pt)
        img_size: Image size as [height, width]
        batch_size: Batch size for export
        dynamic: Enable dynamic ONNX axes
        dynamic_batch: Enable dynamic batch size
        grid: Export Detect() layer grid
        device: Device to use for export
        simplify: Simplify ONNX model using onnxsim
        include_nms: Include NMS in exported model
        opset: ONNX opset version
    
    Returns:
        Path to the exported ONNX file
    """
    import onnx
    from ..models import common
    
    set_logging()
    t = time.time()
    
    # Load PyTorch model
    device = select_device(device)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    labels = model.names
    
    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
    
    # Input
    img = torch.zeros(batch_size, 3, *img_size).to(device)
    img_ir = torch.zeros(batch_size, 3, *img_size).to(device)
    
    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    
    model.model[-1].export = not grid  # set Detect() layer grid export
    
    y = model(img)  # dry run
    if include_nms:
        model.model[-1].include_nms = True
        y = None
    
    # Wrap model with FakeRgbt
    model = FakeRgbt(model)
    
    # Prepare export
    f = weights.replace('.pt', '_fake_rgbt.onnx')  # output filename
    model.eval()
    output_names = ['classes', 'boxes'] if y is None else ['output']
    
    # Configure dynamic axes
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'rgb': {0: 'batch', 2: 'height', 3: 'width'},
            'ir': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'y', 3: 'x'}
        }
    if dynamic_batch:
        batch_size = 'batch'
        dynamic_axes = {
            'rgb': {0: 'batch'},
            'ir': {0: 'batch'},
        }
        if include_nms:
            dynamic_axes.update({
                'num_dets': {0: 'batch'},
                'det_boxes': {0: 'batch'},
                'det_scores': {0: 'batch'},
                'det_classes': {0: 'batch'},
            })
        else:
            dynamic_axes.update({'output': {0: 'batch'}})
    
    if grid:
        if include_nms:
            print('End2end mode is not currently supported in this configuration')
        else:
            model.rgb.model[-1].concat = True
    
    # Export to ONNX
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(
        model, 
        (img, img_ir), 
        f, 
        verbose=False, 
        opset_version=opset, 
        input_names=['rgb', 'ir'],
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    
    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            print('\nStarting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
            print('ONNX simplified successfully')
        except Exception as e:
            print(f'Simplifier failure: {e}')
    
    print('ONNX export success, saved as %s' % f)
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
    
    return f


def export_fake_rgbt_command():
    """
    Command line entry point for exporting models to Fake RGBT ONNX.
    
    This function is called when the user runs 'export-fake-rgbt' command.
    It parses command line arguments and exports the model to ONNX format.
    """
    opt = parse_opt()
    
    # Expand img_size if single value
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    
    export_fake_rgbt(
        weights=opt.weights,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        dynamic=opt.dynamic,
        dynamic_batch=opt.dynamic_batch,
        grid=opt.grid,
        device=opt.device,
        simplify=opt.simplify,
        include_nms=opt.include_nms,
        opset=opt.opset
    )
