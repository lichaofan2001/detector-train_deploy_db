"""
Export ONNX with scaled confidence CLI for detector package.

This module provides the command line interface for exporting PyTorch models to ONNX format
with scaled confidence values.
"""

import sys
import time
import torch
import torch.nn as nn

from detector.models.experimental import attempt_load, End2End
from detector.utils.general import set_logging, check_img_size
from detector.utils.torch_utils import select_device
from detector.utils.activations import Hardswish, SiLU
from detector.utils.add_nms import RegisterNMS


class GridConfModel(nn.Module):
    """Model wrapper that scales confidence for grid mode export."""
    
    def __init__(self, model, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.conf = conf 
        self.x_model = model 
        
    def forward(self, img):
        y = self.x_model(img)
        
        # Scale the confidence
        y[:, :, 4] *= self.conf 
        
        return y 


def retransform(x, s):
    """Retransform confidence values for non-grid mode."""
    return -torch.log((1.0 - s + torch.exp(-x)) / s)


class NoGridConfModel(nn.Module):
    """Model wrapper that scales confidence for non-grid mode export."""
    
    def __init__(self, model, scale_conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.scale_conf = scale_conf 
        self.x_model = model 
        
    def forward(self, img):
        y = self.x_model(img)
        nc = self.x_model.model[-1].nc 
        
        # Scale confidence for each detection layer output
        for i in range(3):
            temp = y[i]
            xywh, conf, class_conf = temp.split((4, 1, nc), 4)
            conf = retransform(conf, self.scale_conf)
            y[i] = torch.cat((xywh, conf, class_conf), 4) 
        
        return y 


def parse_opt():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='导出带缩放置信度的 PyTorch 模型到 ONNX 格式')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1024, 1024], help='图像尺寸（高，宽）')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--dynamic', action='store_true', help='动态 ONNX 轴')
    parser.add_argument('--dynamic-batch', action='store_true', help='动态批次大小（用于 TensorRT 和 ONNX Runtime）')
    parser.add_argument('--grid', action='store_true', help='导出 Detect() 层网格')
    parser.add_argument('--end2end', action='store_true', help='导出端到端 ONNX')
    parser.add_argument('--scale-conf', type=float, default=1, help='置信度缩放因子')
    parser.add_argument('--device', default='cpu', help='CUDA 设备，如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--simplify', action='store_true', help='简化 ONNX 模型')
    parser.add_argument('--include-nms', action='store_true', help='导出带 NMS 的端到端 ONNX')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset 版本')
    parser.add_argument('--max-wh', type=int, default=None, help='TensorRT NMS 为 None，ONNX Runtime NMS 为整数值')
    parser.add_argument('--topk-all', type=int, default=100, help='每张图像的前 K 个目标')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS 的 IOU 阈值')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='NMS 的置信度阈值')
    return parser.parse_args()


def export_scale_conf(weights, img_size, batch_size, dynamic=False, dynamic_batch=False,
                      grid=False, end2end=False, scale_conf=1.0, device='cpu', 
                      simplify=False, include_nms=False, opset=12,
                      max_wh=None, topk_all=100, iou_thres=0.45, conf_thres=0.25):
    """
    Export PyTorch model to ONNX format with scaled confidence.
    
    Args:
        weights: Path to the model weights file (.pt)
        img_size: Image size as [height, width]
        batch_size: Batch size for export
        dynamic: Enable dynamic ONNX axes
        dynamic_batch: Enable dynamic batch size
        grid: Export Detect() layer grid
        end2end: Export end-to-end model
        scale_conf: Confidence scale factor
        device: Device to use for export
        simplify: Simplify ONNX model using onnxsim
        include_nms: Include NMS in exported model
        opset: ONNX opset version
        max_wh: None for tensorrt nms, int value for onnx-runtime nms
        topk_all: TopK objects for every image
        iou_thres: IoU threshold for NMS
        conf_thres: Confidence threshold for NMS
    
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
    
    # Prepare output filename
    f = weights.replace('.pt', '.onnx')
    
    if grid:
        prefix = 'npu'
    else:
        prefix = 'gpu'
        
    size_str = '_'.join([f'{x}' for x in img_size])
    f = f[:-5] + f'{prefix}_{size_str}_scale_conf_{scale_conf:.2f}.onnx'
    
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
        if end2end and max_wh is None:
            output_axes = {
                'num_dets': {0: 'batch'},
                'det_boxes': {0: 'batch'},
                'det_scores': {0: 'batch'},
                'det_classes': {0: 'batch'},
            }
        else:
            output_axes = {'output': {0: 'batch'}}
        dynamic_axes.update(output_axes)
    
    if grid:
        if end2end:
            print('\nStarting export end2end onnx model for %s...' % ('TensorRT' if max_wh is None else 'onnxruntime'))
            model = End2End(model, topk_all, iou_thres, conf_thres, max_wh, device, len(labels))
            if end2end and max_wh is None:
                output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                shapes = [batch_size, 1, batch_size, topk_all, 4,
                          batch_size, topk_all, batch_size, topk_all]
            else:
                output_names = ['output']
        else:
            model.model[-1].concat = True
            
        model = GridConfModel(model, scale_conf)
    else:
        model = NoGridConfModel(model, scale_conf)
    
    # Export to ONNX
    export_success = False
    try:
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
        
        # Checks
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        
        if end2end and max_wh is None:
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))
        
        if simplify:
            try:
                import onnxsim
                print('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
                print('ONNX simplified successfully')
            except Exception as e:
                print(f'Simplifier failure: {e}')
        
        onnx.save(onnx_model, f)
        
        print('ONNX export success, saved as %s' % f)
        export_success = True
        
        if include_nms:
            print('Registering NMS plugin for ONNX...')
            mo = RegisterNMS(f)
            mo.register_nms()
            mo.save(f)
        
    except Exception as e:
        print('ONNX export failure: %s' % e)
    
    if export_success:
        print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
    
    return f


def export_scale_conf_command():
    """
    Command line entry point for exporting models with scaled confidence.
    
    This function is called when the user runs 'export-scale-conf' command.
    It parses command line arguments and exports the model to ONNX format.
    """
    opt = parse_opt()
    
    # Expand img_size if single value
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    
    export_scale_conf(
        weights=opt.weights,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        dynamic=opt.dynamic,
        dynamic_batch=opt.dynamic_batch,
        grid=opt.grid,
        end2end=opt.end2end,
        scale_conf=opt.scale_conf,
        device=opt.device,
        simplify=opt.simplify,
        include_nms=opt.include_nms,
        opset=12,
        max_wh=opt.max_wh,
        topk_all=opt.topk_all,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres
    )
