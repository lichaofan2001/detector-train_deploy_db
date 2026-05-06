"""
批量推理命令行接口

该模块提供 detector-predict 命令，用于对图像目录进行批量检测，
输出 YOLO 格式的检测结果文件。
"""

import argparse
import os
from pathlib import Path

from detector.predict_bbox import test, generate_test_list_with_folder, generate_test_list_with_home


def is_img(filename: str) -> bool:
    """检查文件是否为图像文件"""
    exts = ['jpg', 'png', 'jpeg', 'bmp']
    return any(filename.lower().endswith(ext) for ext in exts)


def predict_command():
    """
    批量推理命令行入口点
    
    该工具用于对图像目录进行批量检测，输出 YOLO 格式的检测结果文件。
    支持两种模式：
    1. 单文件夹模式：使用 --voc 指定单个图像文件夹
    2. 多文件夹模式：使用 --home 指定包含多个图像文件夹的根目录
    
    示例用法：
        # 单文件夹模式
        detector-predict --weights runs/train/exp/weights/best.pt \\
            --voc /path/to/images \\
            --batch-size 32 \\
            --img-size 1024 \\
            --conf-thres 0.01 \\
            --iou-thres 0.45
        
        # 多文件夹模式
        detector-predict --weights runs/train/exp/weights/best.pt \\
            --home /path/to/data_home \\
            --batch-size 32 \\
            --img-size 1024
    """
    parser = argparse.ArgumentParser(
        prog='detector-predict',
        description='批量推理工具 - 对图像目录进行批量检测，输出 YOLO 格式结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 单文件夹模式
  detector-predict --weights best.pt --voc ./data/images --batch-size 32

  # 多文件夹模式  
  detector-predict --weights best.pt --home ./data/ --batch-size 32
        '''
    )
    
    # 模型参数
    parser.add_argument('--weights', type=str, default='weights/best.pt', 
                        help='模型权重文件路径 (.pt)')
    
    # 输入模式参数（二选一）
    parser.add_argument('--home', type=str, default=None, 
                        help='包含多个图像文件夹的根目录（多文件夹模式）')
    parser.add_argument('--voc', type=str, default=None, 
                        help='单个图像文件夹路径（单文件夹模式）')
    
    # 推理参数
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='每批次的图像数量，默认 64')
    parser.add_argument('--img-size', type=int, default=1024, 
                        help='推理图像尺寸（像素），默认 1024')
    parser.add_argument('--conf-thres', type=float, default=0.01, 
                        help='目标置信度阈值，低于此值的检测框将被过滤，默认 0.01')
    parser.add_argument('--iou-thres', type=float, default=0.45, 
                        help='NMS（非极大值抑制）的 IOU 阈值，默认 0.45')

    # 输出参数
    parser.add_argument('--save-dir', type=str, default=None, 
                        help='检测结果保存目录，默认为图像目录下的 yolo-rgb 文件夹')
    parser.add_argument('--rgbname', type=str, default='images', 
                        help='图像文件夹名称，默认为 images')
    
    # 推理选项
    parser.add_argument('--augment', action='store_true', 
                        help='使用增强推理模式（多尺度推理）')
    
    opt = parser.parse_args()
    
    # 验证输入参数
    if opt.home is None and opt.voc is None:
        parser.error('必须指定 --home 或 --voc 参数之一')
    
    # 加载类别名称
    wt_dir, _ = os.path.split(opt.weights)
    classname_file = os.path.join(wt_dir, 'classnames.txt')
    
    if os.path.exists(classname_file):
        classnames = []
        with open(classname_file) as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    classnames.append(line)
    else:
        raise Exception(f'错误：未找到类别名称文件：{classname_file}')
    
    # 生成测试列表
    data = 'test_rgb.txt'
    
    if opt.home is None:
        # 单文件夹模式
        generate_test_list_with_folder(opt.voc, data, opt.rgbname)
    else:
        # 多文件夹模式
        generate_test_list_with_home(opt.home, data, opt.rgbname)
    
    # 确定保存目录
    if opt.save_dir is None:
        if opt.home is None:
            opt.save_dir = os.path.join(opt.voc, 'yolo-rgb')
        else:
            opt.save_dir = os.path.join(opt.home, 'yolo-rgb')
    
    # 确定前缀路径
    prefix_path = opt.voc if opt.home is None else opt.home
    
    # 执行测试
    test(
        data,
        opt.weights,
        opt.batch_size,
        opt.img_size,
        opt.conf_thres,
        opt.iou_thres,
        opt.augment,
        half_precision=False,
        trace=False,
        save_dir=opt.save_dir,
        prefix_path=prefix_path,
        classnames=classnames
    )


if __name__ == '__main__':
    predict_command()
