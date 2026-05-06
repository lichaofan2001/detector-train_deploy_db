"""
Analyze List CLI for detector package.

This module provides the command line interface for analyzing false positives
and false negatives in object detection results.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from detector.models.experimental import attempt_load
from detector.utils.datasets import create_dataloader
from detector.utils.general import check_img_size, non_max_suppression, scale_coords, \
    xyxy2xywh, xywh2xyxy, set_logging, colorstr
from detector.utils.metrics import ConfusionMatrix
from detector.utils.plots import plot_one_box
from detector.utils.torch_utils import select_device, time_synchronized, TracedModel
from detector.utils.tools import get_relative_path, imgfile2detfile, split_path


def img2label_path(x):
    """Define label paths as a function of image paths."""
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return 'txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1))


def read_targets(img_path, shape=None):
    """
    Read ground truth labels and boxes from label file.
    
    Args:
        img_path: Path to image file
        shape: Image shape (height, width), if None will read image to get shape
    
    Returns:
        labels: List of class labels
        boxes: List of bounding boxes in pixel coordinates
    """
    lbfile = img2label_path(img_path)
    if shape is not None:
        HH, WW = shape 
    else:
        im = cv2.imread(img_path)
        HH, WW = im.shape[:2]
        
    labels = []
    boxes = []
    if os.path.exists(lbfile):
        with open(lbfile, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    cls, *box = line.strip().split()
                    box = [float(x) for x in box]
                    # xywh2xyxy
                    box = [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2]
                    box = [box[0]*WW, box[1]*HH, box[2]*WW, box[3]*HH]
                    cls = int(cls)
                    
                    labels.append(cls)
                    boxes.append(box)
    return labels, boxes


def analyze_list(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,  # for NMS
         iou_thres=0.6,  # for NMS
         augment=False,
         half_precision=True,
         trace=False,
         device='0',
         uppper_level=5,
         save_dir=None,
         prefix_path=None,
         draw=False,
         compare=False,
         draw_path=None,
         write_tp=False,
         error_list=None):
    """
    Run detection on image list and analyze false positives/negatives.
    
    Args:
        data: Path to txt data list
        weights: Path to model weights file
        batch_size: Batch size for inference
        imgsz: Image size for inference
        conf_thres: Confidence threshold for NMS
        iou_thres: IoU threshold for NMS
        augment: Use augmented inference
        half_precision: Use half precision (FP16)
        trace: Trace model for optimization
        device: Device to use for inference
        uppper_level: Upper level for path splitting
        save_dir: Directory to save detection results
        prefix_path: Prefix path for relative path computation
        draw: Draw detection results on images
        compare: Compare with ground truth in drawings
        draw_path: Path to save drawn images
        write_tp: Write true positive results
        error_list: Path to save error list file
    
    Returns:
        None
    """
    if draw and draw_path is None:
        print('draw_path is None')
        draw = False  # DON'T Draw
    
    compare = draw and compare 
    if save_dir is None and not draw:
        print('Do nothing')
        return 
    
    set_logging()
    device = select_device(device, batch_size=batch_size)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    
    if trace:
        model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()

    if not os.path.exists(data):
        print(f'Error: data file does not exist: {data}')
        exit(-1)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    dataloader = create_dataloader(data, imgsz, batch_size, gs, pad=0.5, rect=True,
                                    prefix=colorstr(f'detect: '))[0]
    
    seen = 0
    
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    
    if error_list is not None and os.path.exists(error_list):
        os.remove(error_list)

    num_imgs = 0

    t0, t1 = 0, 0
    for batch_i, (img, _, paths, shapes) in enumerate(tqdm(dataloader, desc='test')):
        num_imgs += img.shape[0]
        
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, _ = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=None, multi_label=True)
            t1 += time_synchronized() - t

        for si, pred in enumerate(out):
            
            conf_matrix = ConfusionMatrix(nc=len(names), conf=conf_thres, iou_thres=0.5)
            
            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            path = Path(paths[si])  # image path
            
            # Write detection results
            if save_dir:

                txt_file, txt_path = imgfile2detfile(str(path), uppper_level=uppper_level, save_dir=save_dir, prefix_path=prefix_path)
            
                if not os.path.exists(txt_path):
                    os.makedirs(txt_path)
                with open(txt_file, 'w') as f:
                    pass 
            else:
                txt_file = None

            seen += 1
            
            img_path = str(path)
            
            gt_labels, gt_boxes = read_targets(img_path=img_path, shape=shapes[si][0])
            
            # Update confusion matrix
            gt_2labels = [(cls, *box) for cls, box in zip(gt_labels, gt_boxes)]
            
            gt_2labels = torch.Tensor(gt_2labels).to(predn.device)
            
            if len(predn) == 0:
                conf_matrix.batch_empty_pred(gt_labels)
            elif len(gt_labels) == 0:
                conf_matrix.batch_empty_target(predn)
            else:
                conf_matrix.process_batch(predn, gt_2labels)
                
            # Distinguish correct and incorrect predictions
            matrix = conf_matrix.matrix
            
            # Distinguish three folders: correct, false positive, false negative
            tp_num = int(matrix.diagonal().sum())
            fp_num = int((matrix[:-1, :].sum(1) - matrix[:-1, :-1].diagonal()).sum())  # false positive
            fn_num = int((matrix[:, :-1].sum(0) - matrix[:-1, :-1].diagonal()).sum())  # false negative
            
            if (fp_num > 0 or fn_num > 0) and (error_list is not None):
                with open(error_list, 'a', encoding='utf-8') as f:
                    f.write(img_path)
                    f.write('\n')
            
            if draw:
                if prefix_path is not None:
                    rel_path = get_relative_path(img_path, prefix_path)
                else:
                    _, rel_path = split_path(img_path, uppper_level)
                
                im = cv2.imread(img_path)
                
                key_names = []
                
                if fp_num == 0 and fn_num == 0 and write_tp:
                    key_names.append('right')
                else:
                    # Same image may have both fn and fp
                    if fn_num > 0:
                        key_names.append('fn')
                    if fp_num > 0:
                        key_names.append('fp')
                ouput_path_list = [os.path.join(draw_path, key, rel_path) for key in key_names]
                
                text_infov = 'GT={} TP={} FP={} FN={} PD={}'.format(len(gt_labels), tp_num, fp_num, fn_num, len(predn)) 
                
            else:
                ouput_path_list = None
                im = None              
                gt_labels = None 
                gt_boxes = None
                text_infov = None
            
            if ouput_path_list is not None:
                if compare:
                    # Draw ground truth boxes
                    for cls, box in zip(gt_labels, gt_boxes):
                        label = names[cls]
                        plot_one_box(box, im, label=f'[GT] {label}', color=[0, 0, 255], label_down=True)
                                    
            for *xyxy, conf, cls in predn.tolist():
                
                label = f'{names[int(cls)]}'
                if txt_file is not None:
                    line = (conf, *xyxy)  # label format

                    with open(txt_file, 'a') as f:
                        f.write(f'{label} ' + ('%g ' * len(line)).rstrip() % line + '\n')
                if ouput_path_list is not None:
                    # Draw detection boxes
                    plot_one_box(xyxy, im, label=f'{label}@{conf:.2f}', color=[255, 0, 0])
            
            if ouput_path_list is not None:
                cv2.putText(im, text_infov, (0, 20), 0, 3 / 4, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
                for out_path in ouput_path_list:        
                    out_dir, _ = os.path.split(out_path)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    cv2.imwrite(out_path, im)


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='analyze_list.py', description='分析检测结果中的假阳性和假阴性')
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/5cls_6/weights/best.pt', help='模型权重文件路径 (model.pt)')
    parser.add_argument('--data', type=str, required=True, help='数据列表 txt 文件路径')
    parser.add_argument('--batch-size', type=int, default=32, help='每个图像批次的大小')
    parser.add_argument('--img-size', type=int, default=1024, help='推理图像尺寸（像素）')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='目标置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS 的 IOU 阈值')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--no-trace', action='store_true', help='不 trace 模型')
    parser.add_argument('--save-dir', type=str, default=None, help='保存结果的目录路径')
    parser.add_argument('--prefix-path', type=str, default=None, help='数据文件前缀路径')
    parser.add_argument('--draw-path', type=str, default='here', help='保存绘制图像的路径')
    parser.add_argument('--no-draw', default=False, action='store_true', help='不绘制图像')
    parser.add_argument('--no-compare', default=False, action='store_true', help='不在图像中绘制真实标签')
    parser.add_argument('--write-tp', default=False, action='store_true', help='绘制正确预测并保存真阳性图像')
    parser.add_argument('--error-list', default=None, type=str, help='保存错误列表文件的路径（包含误预测图像）')
    return parser.parse_args()


def analyze_list_command():
    """
    Command line entry point for analyzing detection results.
    
    This function is called when the user runs 'detector-analyze' command.
    It parses command line arguments and analyzes false positives/negatives.
    """
    opt = parse_opt()
    print(opt)
    
    analyze_list(
        data=opt.data,
        weights=opt.weights,
        batch_size=opt.batch_size,
        imgsz=opt.img_size,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        augment=opt.augment,
        half_precision=False,
        trace=not opt.no_trace,
        save_dir=opt.save_dir,
        prefix_path=opt.prefix_path,
        draw=not opt.no_draw,
        compare=not opt.no_compare,
        draw_path=opt.draw_path,
        write_tp=opt.write_tp,
        error_list=opt.error_list
    )
