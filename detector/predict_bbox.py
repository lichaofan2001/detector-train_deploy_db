import argparse
import json
import os
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from detector.models.experimental import attempt_load
from detector.utils.datasets import create_dataloader
from detector.utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from detector.utils.metrics import ap_per_class, ConfusionMatrix
from detector.utils.plots import plot_images, output_to_target, plot_study_txt
from detector.utils.torch_utils import select_device, time_synchronized, TracedModel
import json
from detector.utils.tools import imgfile2detfile
# 输出结果按照 yolo 格式书写

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,  # NMS 置信度阈值
         iou_thres=0.6,     # NMS IOU 阈值
         augment=False,
         half_precision=True,
         trace=False,
         device='0',
         uppper_level=3,    # 目录层级级别
         save_dir=None,     # 结果保存目录
         prefix_path=None,  # 输出文件前缀路径
         classnames=None,   # 类别名称列表
         save_images=False, # 是否保存带检测框的图片
         images_dir=None,
         workers=0,
         progress_callback=None,
         ignore_classes=None):  # 忽略的类别ID列表


    set_logging()
    import torch
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'CUDA 不可用，自动切换到 CPU')
        device = 'cpu'
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

    if(not os.path.exists(data)):
        print(f'错误，数据不存在')
        exit(-1)




    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    dataloader = create_dataloader(data, imgsz, batch_size, gs, pad=0.5, rect=True,
                                    prefix=colorstr(f'detect: '), workers=workers)[0]
    
    seen = 0
    

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    if(classnames is None):
        classnames = names

    num_imgs = 0

    t0, t1 = 0, 0
    for batch_i, (img, _, paths, shapes) in enumerate(tqdm(dataloader, desc='test')):
        num_imgs += img.shape[0]
        #num_bboxes += targets.shape[0]
        
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, _ = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t


            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=None, multi_label=True, agnostic=True)
            t1 += time_synchronized() - t

        for si, pred in enumerate(out):
            
            
            path = Path(paths[si])

            # 写入
            # txt_file, txt_path = imgfile2detfile(str(path), uppper_level, save_dir, prefix_path = prefix_path,det_path_name=None)
            txt_file, txt_path = imgfile2detfile(str(path), uppper_level=uppper_level, save_dir=save_dir,prefix_path=prefix_path, det_path_name=None)
            if(not os.path.exists(txt_path)):
                os.makedirs(txt_path)
            with open(txt_file, 'w') as f:
                pass 

            seen += 1

            if len(pred) == 0:
                continue

            # Predictions
            predn = pred.clone()
            
            imgH, imgW = shapes[si][0]
            #print(imgH, imgW)
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file

            for *xyxy, conf, cls in predn.tolist():
                cls_int = int(cls)
                # 检查是否需要忽略该类别
                if ignore_classes and cls_int in ignore_classes:
                    continue

                label = f'{classnames[cls_int]}'
                if(label == 'NaN'):
                    continue # 如果为 NaN，则忽略

                cx = (xyxy[0] + xyxy[2])/2.0/imgW
                cy = (xyxy[1] + xyxy[3])/2.0/imgH
                w = (xyxy[2] - xyxy[0])/imgW
                h = (xyxy[3] - xyxy[1])/imgH

                line = (cx, cy, w, h, conf) # label format

                with open(txt_file, 'a') as f:
                    f.write(f'{label} '+('%g ' * len(line)).rstrip() % line + '\n')

            if save_images and images_dir and len(predn) > 0:
                orig_img_path = str(path)
                if os.path.exists(orig_img_path):
                    img0 = cv2.imread(orig_img_path)
                    if img0 is not None:
                        os.makedirs(images_dir, exist_ok=True)
                        img_name = os.path.basename(orig_img_path)
                        for *box, conf, cls in predn.tolist():
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label_cls = classnames[int(cls)] if int(cls) < len(classnames) else str(int(cls))
                            cv2.putText(img0, f'{label_cls}:{conf:.2f}', (x1, max(20, y1 - 10)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        save_path = os.path.join(images_dir, img_name)
                        cv2.imwrite(save_path, img0)

                        if progress_callback:
                            det_count = len(predn)
                            progress_callback(save_path, det_count)

def is_img(filename:str):
    exts = ['jpg', 'png', 'jpeg', 'bmp']
    for ext in exts:
        if(filename.lower().endswith(ext)):
            return True 
    return False 
# 根据测试图像目录，生成图像列表，进行测试，
def generate_test_list_with_folder(voc, filename='test_rgbt.txt', rgbname='images', mode='w'):
    
    cache_file = filename[:-4] + '.cache'
    if(os.path.exists(cache_file)):
        os.remove(cache_file)
        
    imgfolder = os.path.join(voc, rgbname)
    
    with open(filename, mode, encoding='utf-8') as f:
        for filename in os.listdir(imgfolder):
            if(is_img(filename)):
                fn = os.path.join(imgfolder, filename)

                f.write(fn)
                f.write('\n')
    
def get_voc_list(home, rgbname='images'):
    for pn, _, _ in os.walk(home):
        imgfolder = os.path.join(pn, rgbname)
        
        if(os.path.exists(imgfolder)):
            yield pn
             
def generate_test_list_with_home(home, filename='test_rgbt.txt', rgbname='images'):
    if(os.path.exists(filename)):
        os.remove(filename)
        
    for voc in get_voc_list(home, rgbname):
        print(voc)
        generate_test_list_with_folder(voc, filename, rgbname=rgbname, mode='a')


def predict_command():
    """
    批量推理命令行入口点
    
    该工具用于对图像目录进行批量检测，输出 YOLO 格式的检测结果文件。
    支持两种模式：
    1. 单文件夹模式：使用 --voc 指定单个图像文件夹
    2. 多文件夹模式：使用 --home 指定包含多个图像文件夹的根目录
    """
    parser = argparse.ArgumentParser(prog='detector-predict')
    
    parser.add_argument('--weights', type=str, default='weights/best.pt',
                        help='模型权重文件路径 (.pt)')
    parser.add_argument('--home', type=str, default=None,
                        help='包含多个图像文件夹的根目录（多文件夹模式）')
    parser.add_argument('--voc', type=str, default=None,
                        help='单个图像文件夹路径（单文件夹模式）')
        
    parser.add_argument('--batch-size', type=int, default=64,
                        help='每批次的图像数量')
    parser.add_argument('--img-size', type=int, default=1024,
                        help='推理图像尺寸（像素）')
    parser.add_argument('--conf-thres', type=float, default=0.01,
                        help='目标置信度阈值，低于此值的检测框将被过滤')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='NMS（非极大值抑制）的 IOU 阈值')

    parser.add_argument('--save-dir', type=str, default=None,
                        help='检测结果保存目录，默认为图像目录下的 yolo-rgb 文件夹')
    parser.add_argument('--rgbname', type=str, default='images',
                        help='图像文件夹名称，默认为 images')
    parser.add_argument('--augment', action='store_true',
                        help='使用增强推理模式（多尺度推理）')
    
    

    opt = parser.parse_args()
    print(opt)
    
    # classnames 的文件
    wt_dir, _ = os.path.split(opt.weights)
    classname_file = os.path.join(wt_dir, 'classnames.txt')
    if(os.path.exists(classname_file)):
        classnames = []
        with open(classname_file) as f:
            for line in f.readlines():
                line = line.strip()
                if(line):
                    classnames.append(line)
    else:
        classnames = []
        for i in range(80):
            classnames.append(str(i))
    
    data = 'test_rgb.txt'

    
    
    if(opt.home is None):
        generate_test_list_with_folder(opt.voc, data, opt.rgbname)
    else:
        generate_test_list_with_home(opt.home, data, opt.rgbname)
    
    if(opt.save_dir is None):
        if(opt.home is None):
            opt.save_dir = os.path.join(opt.voc, 'yolo-rgb')
        else:
            opt.save_dir = os.path.join(opt.home, 'yolo-rgb')
    
    if(opt.home is None):
        prefix_path = opt.voc
    else:
        prefix_path = opt.home
        
    #print(data)
    test(data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.augment,
            half_precision=False,
            trace=False,
            save_dir=opt.save_dir,
            prefix_path=prefix_path, classnames=classnames
            )


if __name__ == '__main__':
    predict_command()
