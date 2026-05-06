#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 PT 权重加载模型并对单张图像进行推理的示例代码

功能：
1. 从 .pt 权重文件加载模型
2. 打印模型各层的名称
3. 对单张图像进行推理
4. 输出检测框结果

使用方法：
    python examples/inference_example.py --weights best.pt --image test.jpg --conf-thres 0.25
"""

import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path

# 导入 detector 模块
from detector.models.experimental import attempt_load
from detector.utils.general import non_max_suppression, scale_coords, check_img_size
from detector.utils.datasets import letterbox
from detector.utils.torch_utils import select_device, time_synchronized


def load_model(weights_path, device='0'):
    """
    从 .pt 权重文件加载模型
    
    Args:
        weights_path: 权重文件路径 (.pt)
        device: 推理设备，如 '0'、'0,1' 或 'cpu'
    
    Returns:
        model: 加载的模型
        device: torch.device 对象
    """
    # 选择设备
    device = select_device(device)
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {weights_path}")
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    
    # 获取模型信息
    stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
    names = model.module.names if hasattr(model, 'module') else model.names
    print(f"模型步长: {stride}")
    print(f"类别数量: {len(names)}")
    print(f"类别名称: {names}")
    
    return model, device, stride, names


def print_model_layers(model):
    """
    打印模型各层的名称和类型
    
    Args:
        model: PyTorch 模型
    """
    print("\n" + "=" * 80)
    print("模型各层名称和类型")
    print("=" * 80)
    
    # 方式一：遍历所有模块
    print("\n【方式一】named_modules() - 所有子模块:")
    print("-" * 80)
    for name, module in model.named_modules():
        if name:  # 跳过根模块（空名称）
            print(f"  {name}: {type(module).__name__}")
    
    # 方式二：遍历所有参数
    print("\n【方式二】named_parameters() - 所有参数:")
    print("-" * 80)
    param_count = 0
    for name, param in model.named_parameters():
        param_count += 1
        print(f"  {name}: shape={list(param.shape)}, dtype={param.dtype}")
    print(f"  总参数数量: {param_count}")
    
    # 方式三：遍历模型的 model 属性（如果存在）
    if hasattr(model, 'model'):
        print("\n【方式三】model.model 层列表:")
        print("-" * 80)
        for i, m in enumerate(model.model):
            print(f"  [{i:2d}] {type(m).__name__}: {m}")
    
    # 计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print("=" * 80 + "\n")


def preprocess_image(image_path, img_size=640, stride=32, device='cpu'):
    """
    图像预处理
    
    Args:
        image_path: 图像文件路径
        img_size: 目标图像尺寸
        stride: 模型步长
        device: 推理设备
    
    Returns:
        img: 预处理后的张量
        img0: 原始图像
        ratio: 缩放比例
        (dw, dh): 填充大小
    """
    # 读取图像
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    print(f"原始图像尺寸: {img0.shape[:2]} (H, W)")
    
    # 检查图像尺寸是否为步长的倍数
    img_size = check_img_size(img_size, s=stride)
    
    # 使用 letterbox 进行填充缩放
    img = letterbox(img0, new_shape=img_size, stride=stride)[0]
    
    # BGR to RGB, HWC to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    
    # 转换为张量
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 归一化到 0-1
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 添加 batch 维度
    
    print(f"预处理后张量尺寸: {img.shape}")
    
    return img, img0


def inference(model, img, conf_thres=0.25, iou_thres=0.45, augment=False):
    """
    模型推理
    
    Args:
        model: 模型
        img: 预处理后的图像张量
        conf_thres: 置信度阈值
        iou_thres: IoU 阈值
        augment: 是否使用增强推理
    
    Returns:
        pred: NMS 后的预测结果
        inference_time: 推理时间 (ms)
        nms_time: NMS 时间 (ms)
    """
    # 推理
    t1 = time_synchronized()
    with torch.no_grad():
        pred = model(img, augment=augment)[0]
    t2 = time_synchronized()
    
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    t3 = time_synchronized()
    
    inference_time = (t2 - t1) * 1000
    nms_time = (t3 - t2) * 1000
    
    return pred, inference_time, nms_time


def process_detections(pred, img_shape, img0_shape, names):
    """
    处理检测结果
    
    Args:
        pred: NMS 后的预测结果
        img_shape: 预处理后图像尺寸 (C, H, W)
        img0_shape: 原始图像尺寸 (H, W, C)
        names: 类别名称列表
    
    Returns:
        detections: 检测结果列表
    """
    detections = []
    
    for det in pred:
        if len(det):
            print(det, img_shape[2:], img0_shape)
            # 将坐标从预处理图像缩放回原图
            det[:, :4] = scale_coords(img_shape[2:], det[:, :4], img0_shape).round()

            print('new: ', det)
            
            for *xyxy, conf, cls in det:

                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cls_id = int(cls)
                cls_name = names[cls_id] if cls_id < len(names) else str(cls_id)
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': cls_id,
                    'class_name': cls_name
                }
                detections.append(detection)
    
    return detections


def draw_detections(img0, detections, output_path=None):
    """
    在图像上绘制检测框
    
    Args:
        img0: 原始图像
        detections: 检测结果列表
        output_path: 输出路径
    
    Returns:
        img_with_boxes: 带检测框的图像
    """
    img_with_boxes = img0.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cls_name = det['class_name']
        
        # 随机颜色
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        
        # 绘制边界框
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{cls_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_with_boxes, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(img_with_boxes, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if output_path:
        cv2.imwrite(output_path, img_with_boxes)
        print(f"结果已保存到: {output_path}")
    
    return img_with_boxes


def main():
    parser = argparse.ArgumentParser(description='从 PT 权重加载模型并进行推理')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径 (.pt)')
    parser.add_argument('--image', type=str, required=True, help='图像路径')
    parser.add_argument('--img-size', type=int, default=640, help='推理图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--device', default='0', help='CUDA 设备，如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--augment', action='store_true', help='使用增强推理')
    parser.add_argument('--output', type=str, default=None, help='输出图像路径')
    parser.add_argument('--print-layers', action='store_true', help='打印模型各层名称')
    
    args = parser.parse_args()
    
    # 1. 加载模型
    print("\n" + "=" * 60)
    print("步骤 1: 加载模型")
    print("=" * 60)
    model, device, stride, names = load_model(args.weights, args.device)
    
    # 2. 打印模型层信息（可选）
    if args.print_layers:
        print_model_layers(model)
    
    # 3. 图像预处理
    print("\n" + "=" * 60)
    print("步骤 2: 图像预处理")
    print("=" * 60)
    img, img0 = preprocess_image(args.image, args.img_size, stride, device)
    
    # 4. 模型推理
    print("\n" + "=" * 60)
    print("步骤 3: 模型推理")
    print("=" * 60)
    pred, inf_time, nms_time = inference(
        model, img, args.conf_thres, args.iou_thres, args.augment
    )
    print(f"推理时间: {inf_time:.1f} ms")
    print(f"NMS 时间: {nms_time:.1f} ms")
    print(f"总时间: {inf_time + nms_time:.1f} ms")
    
    # 5. 处理检测结果
    print("\n" + "=" * 60)
    print("步骤 4: 处理检测结果")
    print("=" * 60)
    detections = process_detections(pred, img.shape, img0.shape[:2], names)
    
    print(f"检测到 {len(detections)} 个目标:")
    print("-" * 60)
    for i, det in enumerate(detections):
        print(f"  [{i+1}] {det['class_name']}: 置信度={det['confidence']:.3f}, "
              f"边界框=[{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]")
    
    # 6. 可视化结果
    print("\n" + "=" * 60)
    print("步骤 5: 可视化结果")
    print("=" * 60)
    output_path = args.output or f"result_{Path(args.image).name}"
    draw_detections(img0, detections, output_path)
    
    print("\n完成！")
    
    return detections


def simple_inference_example():
    """
    最简推理示例代码
    
    展示如何在代码中最简单地使用 detector 进行推理
    """
    import torch
    import cv2
    import numpy as np
    from detector.models.experimental import attempt_load
    from detector.utils.general import non_max_suppression, scale_coords, letterbox
    from detector.utils.torch_utils import select_device
    
    # ============ 配置 ============
    weights_path = 'best.pt'        # 权重路径
    image_path = 'test.jpg'         # 图像路径
    img_size = 640                  # 推理尺寸
    conf_thres = 0.25               # 置信度阈值
    iou_thres = 0.45                # IoU 阈值
    device = '0'                    # 设备
    
    # ============ 加载模型 ============
    device = select_device(device)
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names
    
    # ============ 图像预处理 ============
    img0 = cv2.imread(image_path)
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # ============ 推理 ============
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # ============ 处理结果 ============
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                print(f"检测到: {names[int(cls)]}, 置信度: {conf:.2f}, "
                      f"位置: [{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]")


if __name__ == '__main__':
    main()