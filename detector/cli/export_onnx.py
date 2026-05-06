"""
Export ONNX CLI for detector package.

This module provides the command line interface for exporting PyTorch models to ONNX format.
"""

import sys
import time
import torch
import torch.nn as nn

from detector.models.experimental import attempt_load
from detector.utils.general import set_logging, check_img_size
from detector.utils.torch_utils import select_device
from detector.utils.activations import Hardswish, SiLU


def parse_opt():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='导出 PyTorch 模型到 ONNX 格式')
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
    parser.add_argument('--topk-all', type=int, default=100, help='每张图像的前 K 个目标')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS 的 IOU 阈值')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='NMS 的置信度阈值')
    parser.add_argument('--max-wh', type=int, default=None, help='TensorRT NMS 为 None，ONNX Runtime NMS 为整数值')
    return parser.parse_args()


def export_onnx(weights, img_size, batch_size, dynamic=False, dynamic_batch=False,
              grid=False, device='cpu', simplify=False, include_nms=False, opset=12,
              topk_all=100, iou_thres=0.45, conf_thres=0.25, max_wh=None):
    """
    Export PyTorch model to ONNX format.
    
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
        topk_all: TopK objects for every image (for NMS)
        iou_thres: IoU threshold for NMS
        conf_thres: Confidence threshold for NMS
        max_wh: None for tensorrt nms, int value for onnx-runtime nms
    
    Returns:
        Path to the exported ONNX file
    """
    import onnx
    from ..models.experimental import End2End
    from ..models import common, yolo
    
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
    
    # Prepare export
    f = weights.replace('.pt', '.onnx')  # output filename
    model.eval()
    output_names = ['classes', 'boxes'] if y is None else ['output']
    
    # Configure dynamic axes
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'y', 3: 'x'}
        }
    if dynamic_batch:
        batch_size = 'batch'
        dynamic_axes = {'images': {0: 'batch'}}
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
            print('\nStarting export end2end onnx model for %s...' % ('TensorRT' if max_wh is None else 'onnxruntime'))
            model = End2End(model, topk_all, iou_thres, conf_thres, max_wh, device, len(labels))
            if max_wh is None:
                output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                shapes = [batch_size, 1, batch_size, topk_all, 4,
                          batch_size, topk_all, batch_size, topk_all]
            else:
                output_names = ['output']
        else:
            model.model[-1].concat = True
    
    # Export to ONNX
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(
        model,
        img,
        f,
        verbose=False,
        opset_version=opset,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Fix shapes for end2end with max_wh=None
    if grid and include_nms and max_wh is None:
        onnx_model = onnx.load(f)
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))
        onnx.save(onnx_model, f)
    
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
            onnx.save(onnx_model, f)
            print('ONNX simplified successfully')
        except Exception as e:
            print(f'Simplifier failure: {e}')
    
    print('ONNX export success, saved as %s' % f)
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
    
    return f


def export_onnx_command():
    """
    Command line entry point for exporting models to ONNX.
    
    This function is called when the user runs 'export-onnx' command.
    It parses command line arguments and exports the model to ONNX format.
    """
    opt = parse_opt()
    
    # Expand img_size if single value
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    
    export_onnx(
        weights=opt.weights,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        dynamic=opt.dynamic,
        dynamic_batch=opt.dynamic_batch,
        grid=opt.grid,
        device=opt.device,
        simplify=opt.simplify,
        include_nms=opt.include_nms,
        opset=opt.opset,
        topk_all=opt.topk_all,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        max_wh=opt.max_wh
    )
