"""
Detect CLI for detector package.

This module provides the command line interface for running object detection
on images, videos, or streams.
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from detector.models.experimental import attempt_load
from detector.utils.datasets import LoadStreams, LoadImages
from detector.utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, check_requirements
from detector.utils.module_loader import torch_load
from detector.utils.plots import plot_one_box
from detector.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def get_increment_path(opt):
    """Get incremented path for saving results."""
    return Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))


def detect(opt, save_img=False, save_dir=None):
    """
    Run object detection on images, videos, or streams.
    
    Args:
        opt: Parsed command line arguments
        save_img: Whether to save images (default: False)
        save_dir: Directory to save results (default: None, uses opt to generate)
    
    Returns:
        None
    """
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    if save_dir is None:
        save_dir = get_increment_path(opt)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch_load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        if view_img:
            view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow('winname', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv7 目标检测推理脚本')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='模型权重文件路径 (model.pt)')
    parser.add_argument('--source', type=str, default='inference/images', help='检测源（文件/文件夹，0 表示摄像头）')
    parser.add_argument('--img-size', type=int, default=1024, help='推理图像尺寸（像素）')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='目标置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS 的 IOU 阈值')
    parser.add_argument('--device', default='', help='CUDA 设备，如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--view-img', action='store_true', help='显示检测结果')
    parser.add_argument('--save-txt', action='store_true', help='将结果保存到 *.txt 文件')
    parser.add_argument('--save-conf', action='store_true', help='在 --save-txt 标签中保存置信度')
    parser.add_argument('--nosave', action='store_true', help='不保存图像/视频结果')
    parser.add_argument('--classes', nargs='+', type=int, help='按类别过滤：--class 0 或 --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关的 NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--update', action='store_true', help='更新所有模型（修复 SourceChangeWarning）')
    parser.add_argument('--project', default='runs/detect', help='保存结果到 project/name')
    parser.add_argument('--name', default='exp', help='保存结果到 project/name')
    parser.add_argument('--exist-ok', action='store_true', help='已存在的 project/name 不递增')
    parser.add_argument('--no-trace', action='store_true', help='不 trace 模型')
    return parser.parse_args()


def detect_command():
    """
    Command line entry point for running object detection.
    
    This function is called when the user runs 'detector-detect' command.
    It parses command line arguments and runs detection on the specified source.
    """
    opt = parse_opt()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)
