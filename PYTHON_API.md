# Detector Python API 文档

本文档详细描述了 `detector` 包提供的 Python API 接口，包括模型训练、模型评估、推理预测等核心功能的使用方法。

## 目录

- [安装与配置](#安装与配置)
- [快速开始](#快速开始)
- [核心 API](#核心-api)
  - [模型训练](#模型训练)
  - [模型评估](#模型评估)
  - [批量推理](#批量推理)
  - [结果可视化](#结果可视化)
- [辅助函数](#辅助函数)
- [模型加载与推理](#模型加载与推理)
- [完整示例](#完整示例)

---

## 安装与配置

### 安装

```bash
# 开发模式安装
pip install -e .

# 或作为普通包安装
pip install .
```

### 导入模块

```python
# 导入核心功能
from detector import train, test, show_command, predict_command

# 导入路径辅助函数
from detector import (
    get_cfg_path,
    get_data_path,
    get_hyp_path,
    get_ckpoints_path,
    get_default_cfg_path,
    get_default_weights_path,
    get_default_data_path,
    get_default_hyp_path
)

# 导入模型相关
from detector.models.yolo import Model
from detector.models.experimental import attempt_load
```

---

## 快速开始

### 最简训练示例

```python
from detector import train
from detector.utils.torch_utils import select_device
import yaml

# 加载超参数
with open('data/hyp.scratch.tiny.yaml') as f:
    hyp = yaml.safe_load(f)

# 创建配置对象
class Opt:
    weights = 'yolov7-tiny.pt'
    cfg = 'cfg/yolov7-tiny-silu.yaml'
    data = 'data/data.yaml'
    epochs = 100
    batch_size = 16
    img_size = [640, 640]
    device = '0'
    # ... 其他参数

# 选择设备
device = select_device('0')

# 开始训练
train(hyp, Opt(), device)
```

### 最简推理示例

```python
import torch
from detector.models.experimental import attempt_load
from detector.utils.general import non_max_suppression, scale_coords
from detector.utils.datasets import LoadImages
from detector.utils.torch_utils import select_device

# 加载模型
device = select_device('0')
model = attempt_load('best.pt', map_location=device)
model.eval()

# 推理
dataset = LoadImages('path/to/images', img_size=640)
for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device).float() / 255.0
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    # 处理预测结果...
```

---

## 核心 API

---

### 模型训练

#### `detector.train.train()`

训练目标检测模型的主函数。

**函数签名**:

```python
def train(hyp, opt, device, tb_writer=None):
    """
    主训练函数
    
    Args:
        hyp (dict): 超参数字典，包含学习率、动量、权重衰减等
        opt (argparse.Namespace): 训练配置选项
        device (torch.device): 训练设备
        tb_writer (SummaryWriter, optional): TensorBoard 写入器
    
    Returns:
        tuple: 训练结果 (mp, mr, map50, map, box_loss, obj_loss, cls_loss)
    """
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `hyp` | dict | 超参数字典，通常从 YAML 文件加载 |
| `opt` | Namespace | 训练配置选项对象 |
| `device` | torch.device | 训练设备 (CPU/GPU) |
| `tb_writer` | SummaryWriter | 可选的 TensorBoard 日志写入器 |

**`opt` 对象必需属性**:

```python
class TrainOptions:
    # 路径配置
    weights: str = 'yolov7-tiny.pt'      # 预训练权重路径
    cfg: str = 'cfg/yolov7-tiny.yaml'    # 模型配置文件路径
    data: str = 'data/data.yaml'         # 数据集配置文件路径
    hyp: str = 'data/hyp.scratch.tiny.yaml'  # 超参数文件路径
    
    # 训练参数
    epochs: int = 100                     # 训练轮数
    batch_size: int = 16                  # 批次大小
    img_size: list = [640, 640]          # 图像尺寸 [train, test]
    
    # 设备配置
    device: str = '0'                     # CUDA 设备
    workers: int = 8                      # 数据加载线程数
    
    # 优化器配置
    adam: bool = False                    # 使用 Adam 优化器
    linear_lr: bool = False               # 线性学习率衰减
    
    # 训练选项
    rect: bool = False                    # 矩形训练
    resume: bool = False                  # 恢复训练
    nosave: bool = False                  # 仅保存最终检查点
    notest: bool = False                  # 仅测试最终 epoch
    noautoanchor: bool = False            # 禁用自动锚框
    evolve: bool = False                  # 超参数进化
    cache_images: bool = False            # 缓存图像
    image_weights: bool = False           # 图像加权采样
    multi_scale: bool = False             # 多尺度训练
    single_cls: bool = False              # 单类别训练
    
    # 分布式训练
    sync_bn: bool = False                 # 同步 BN
    local_rank: int = -1                  # DDP 本地 rank
    world_size: int = 1                   # DDP world size
    global_rank: int = -1                 # DDP 全局 rank
    
    # 保存配置
    project: str = 'runs/train'           # 项目目录
    name: str = 'exp'                     # 实验名称
    exist_ok: bool = False                # 覆盖已存在目录
    save_period: int = -1                 # 保存周期
    
    # 其他选项
    freeze: list = [0]                    # 冻结层
    quad: bool = False                    # 四路数据加载器
    label_smoothing: float = 0.0          # 标签平滑
    v5_metric: bool = False               # 使用 YOLOv5 指标
```

**`hyp` 超参数字典**:

```python
# 学习率相关
lr0: float = 0.01           # 初始学习率
lrf: float = 0.1            # 最终学习率系数
momentum: float = 0.937     # SGD 动量
weight_decay: float = 0.0005  # 权重衰减

# 预热
warmup_epochs: int = 3      # 预热轮数
warmup_momentum: float = 0.8
warmup_bias_lr: float = 0.1

# 损失权重
box: float = 0.05           # 边界框损失权重
cls: float = 0.5            # 分类损失权重
obj: float = 1.0            # 目标损失权重

# 数据增强
hsv_h: float = 0.015        # HSV-H 增强
hsv_s: float = 0.7          # HSV-S 增强
hsv_v: float = 0.4          # HSV-V 增强
degrees: float = 0.0        # 旋转角度
translate: float = 0.2      # 平移
scale: float = 0.9          # 缩放
shear: float = 0.0          # 剪切
flipud: float = 0.0         # 上下翻转
fliplr: float = 0.5         # 左右翻转
mosaic: float = 1.0         # Mosaic 增强
mixup: float = 0.0          # Mixup 增强

# 锚框
anchor_t: float = 4.0       # 锚框阈值
anchors: int = 3            # 每层锚框数
```

**使用示例**:

```python
import argparse
import yaml
import torch
from detector.train import train
from detector.utils.torch_utils import select_device

# 方式一：使用 argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='yolov7-tiny.pt')
parser.add_argument('--cfg', type=str, default='cfg/yolov7-tiny.yaml')
parser.add_argument('--data', type=str, default='data/data.yaml')
parser.add_argument('--hyp', type=str, default='data/hyp.scratch.tiny.yaml')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640])
parser.add_argument('--device', default='0')
# ... 添加更多参数
opt = parser.parse_args()

# 加载超参数
with open(opt.hyp) as f:
    hyp = yaml.safe_load(f)

# 选择设备
device = select_device(opt.device)

# 开始训练
train(hyp, opt, device)


# 方式二：直接创建配置对象
class Opt:
    weights = 'yolov7-tiny.pt'
    cfg = 'cfg/yolov7-tiny.yaml'
    data = 'data/data.yaml'
    epochs = 100
    batch_size = 16
    img_size = [640, 640]
    device = '0'
    workers = 8
    project = 'runs/train'
    name = 'exp'
    exist_ok = False
    rect = False
    resume = False
    nosave = False
    notest = False
    noautoanchor = False
    evolve = False
    cache_images = False
    image_weights = False
    multi_scale = False
    single_cls = False
    adam = False
    sync_bn = False
    local_rank = -1
    world_size = 1
    global_rank = -1
    freeze = [0]
    quad = False
    linear_lr = False
    label_smoothing = 0.0
    save_period = -1
    v5_metric = False
    
    def __init__(self):
        self.save_dir = f'{self.project}/{self.name}'
        self.total_batch_size = self.batch_size

opt = Opt()
with open('data/hyp.scratch.tiny.yaml') as f:
    hyp = yaml.safe_load(f)
device = select_device('0')
train(hyp, opt, device)
```

**返回值**:

训练完成后返回结果元组：

```python
results = (mp, mr, map50, map, box_loss, obj_loss, cls_loss)
# mp: 平均精度 (Precision)
# mr: 平均召回率 (Recall)
# map50: mAP@0.5
# map: mAP@0.5:0.95
# box_loss: 边界框损失
# obj_loss: 目标损失
# cls_loss: 分类损失
```

---

### 模型评估

#### `detector.test.test()`

测试/评估目标检测模型的函数。

**函数签名**:

```python
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),
         save_txt=False,
         save_hybrid=False,
         save_conf=False,
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         opt_task='test',
         opt_device='0',
         matrix_conf_thresh=0.25,
         opt=None):
    """
    模型测试/评估函数
    
    Args:
        data: 数据集配置文件路径或字典
        weights: 模型权重文件路径
        batch_size: 批次大小
        imgsz: 推理图像尺寸
        conf_thres: NMS 置信度阈值
        iou_thres: NMS IoU 阈值
        save_json: 保存 COCO 格式 JSON 结果
        single_cls: 作为单类别数据集处理
        augment: 增强推理
        verbose: 详细输出每类 mAP
        model: 模型实例（训练时使用）
        dataloader: 数据加载器实例
        save_dir: 结果保存目录
        save_txt: 保存检测结果到 txt 文件
        save_hybrid: 保存混合标签
        save_conf: 保存置信度
        plots: 生成可视化图表
        wandb_logger: W&B 日志记录器
        compute_loss: 损失计算函数
        half_precision: 使用半精度推理
        trace: 追踪模型
        is_coco: 是否为 COCO 数据集
        v5_metric: 使用 YOLOv5 指标
        opt_task: 任务类型 ('train', 'val', 'test')
        opt_device: 推理设备
        matrix_conf_thresh: 混淆矩阵置信度阈值
        opt: 全局配置对象
    
    Returns:
        tuple: (metrics, maps, times)
            - metrics: (mp, mr, map50, map, box_loss, obj_loss, cls_loss)
            - maps: 每类 mAP 数组
            - times: (推理时间, NMS时间, 总时间)
    """
```

**使用示例**:

```python
from pathlib import Path
from detector.test import test
from detector.utils.torch_utils import select_device

# 方式一：独立测试
class TestOpt:
    task = 'val'
    device = '0'
    project = 'runs/test'
    name = 'exp'
    exist_ok = False

results, maps, times = test(
    data='data/data.yaml',
    weights='runs/train/exp/weights/best.pt',
    batch_size=32,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,
    opt_task='val',
    opt_device='0',
    opt=TestOpt()
)

print(f"mAP@0.5: {results[2]:.4f}")
print(f"mAP@0.5:0.95: {results[3]:.4f}")


# 方式二：在训练过程中测试
# 通常在 train.py 中自动调用，传入 model 和 dataloader
results, maps, times = test(
    data=data_dict,
    model=model,
    dataloader=testloader,
    compute_loss=compute_loss,
    half_precision=True,
    plots=True
)


# 方式三：多模型对比测试
weights_list = ['model1.pt', 'model2.pt', 'model3.pt']
for weights in weights_list:
    results, maps, times = test(
        data='data/data.yaml',
        weights=weights,
        imgsz=640
    )
    print(f"{weights}: mAP@0.5 = {results[2]:.4f}")
```

**输出结果**:

```
Class      Images    Labels       P       R   mAP@.5  mAP@.5:.95
all          500      2500    0.85    0.82    0.88        0.65
person       500       800    0.87    0.85    0.90        0.68
car          500      1200    0.83    0.80    0.86        0.62
...
```

---

### 批量推理

#### `detector.predict_bbox.test()`

对图像目录进行批量检测，输出 YOLO 格式的检测结果文件。

**函数签名**:

```python
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,
         augment=False,
         half_precision=True,
         trace=False,
         device='0',
         uppper_level=3,
         save_dir=None,
         prefix_path=None,
         classnames=None):
    """
    批量推理函数
    
    Args:
        data: 图像列表文件路径
        weights: 模型权重文件路径
        batch_size: 批次大小
        imgsz: 推理图像尺寸
        conf_thres: NMS 置信度阈值
        iou_thres: NMS IoU 阈值
        augment: 增强推理
        half_precision: 半精度推理
        trace: 追踪模型
        device: 推理设备
        uppper_level: 目录层级级别
        save_dir: 结果保存目录
        prefix_path: 输出文件前缀路径
        classnames: 类别名称列表
    
    Returns:
        None (结果直接写入文件)
    """
```

**使用示例**:

```python
import os
from detector.predict_bbox import test, generate_test_list_with_folder, generate_test_list_with_home

# 方式一：单文件夹推理
voc = '/path/to/images'
data_list = 'test_images.txt'

# 生成图像列表
generate_test_list_with_folder(voc, data_list, rgbname='images')

# 执行推理
test(
    data=data_list,
    weights='best.pt',
    batch_size=32,
    imgsz=640,
    conf_thres=0.01,
    iou_thres=0.45,
    device='0',
    save_dir=os.path.join(voc, 'yolo-rgb'),
    prefix_path=voc,
    classnames=['person', 'car', 'dog']  # 或从文件加载
)


# 方式二：多文件夹批量推理
home = '/path/to/data_home'
data_list = 'test_all.txt'

generate_test_list_with_home(home, data_list, rgbname='images')

test(
    data=data_list,
    weights='best.pt',
    batch_size=64,
    imgsz=1024,
    conf_thres=0.01,
    save_dir=os.path.join(home, 'yolo-rgb'),
    prefix_path=home
)
```

**输出格式**:

检测结果保存为 YOLO 格式的 txt 文件：

```
# 每行格式：class_name cx cy w h confidence
person 0.512 0.345 0.123 0.234 0.89
car 0.234 0.567 0.156 0.189 0.95
```

---

### 结果可视化

#### `detector.show_detector_results.write_effect()`

将检测结果绘制到图像上。

**函数签名**:

```python
def write_effect(result_dir, jpg_dir, save_dir, conf_thresh=None):
    """
    绘制检测结果到图像
    
    Args:
        result_dir: 检测结果目录（包含 txt 文件）
        jpg_dir: 原始图像目录
        save_dir: 保存目录
        conf_thresh: 置信度阈值过滤
    """
```

**使用示例**:

```python
from detector.show_detector_results import write_effect, show_bbox_prd

# 批量绘制
write_effect(
    result_dir='./results/yolo-rgb',
    jpg_dir='./data/images',
    save_dir='./visualizations',
    conf_thresh=0.3
)

# 单张图像绘制
import cv2
show_bbox_prd(
    im_fp='./data/images/001.jpg',
    lb_fp='./results/yolo-rgb/001.txt',
    dstdir='./output',
    conf_thresh=0.25
)
```

---

## 辅助函数

### 路径辅助函数

```python
from detector import (
    get_package_root,
    get_cfg_path,
    get_data_path,
    get_hyp_path,
    get_detector_home,
    get_ckpoints_path,
    get_default_cfg_path,
    get_default_weights_path,
    get_default_data_path,
    get_default_hyp_path
)

# 获取包根目录
root = get_package_root()
# /path/to/detector

# 获取配置文件路径
cfg_path = get_cfg_path('yolov7-tiny-silu.yaml')
# /path/to/detector/cfg/yolov7-tiny-silu.yaml

# 获取数据配置路径
data_path = get_data_path('data_test.yaml')
# /path/to/detector/data/data_test.yaml

# 获取超参数文件路径
hyp_path = get_hyp_path('hyp.scratch.tiny.yaml')
# /path/to/detector/data/hyp.scratch.tiny.yaml

# 获取用户 checkpoint 目录
ckpoint_dir = get_ckpoints_path()
# ~/.detector/ckpoints

# 获取默认路径
default_cfg = get_default_cfg_path()
default_weights = get_default_weights_path()
default_data = get_default_data_path()
default_hyp = get_default_hyp_path(scratch=True)
```

### 图像列表生成

```python
from detector.predict_bbox import (
    generate_test_list_with_folder,
    generate_test_list_with_home,
    get_voc_list,
    is_img
)

# 检查是否为图像文件
is_img('test.jpg')  # True
is_img('test.txt')  # False

# 生成单文件夹图像列表
generate_test_list_with_folder(
    voc='/path/to/images',
    filename='test.txt',
    rgbname='images',
    mode='w'
)

# 生成多文件夹图像列表
generate_test_list_with_home(
    home='/path/to/data_home',
    filename='test_all.txt',
    rgbname='images'
)

# 获取所有包含图像文件夹的路径
for voc_path in get_voc_list('/path/to/home', rgbname='images'):
    print(voc_path)
```

---

## 模型加载与推理

### 加载模型

```python
import torch
from detector.models.experimental import attempt_load
from detector.models.yolo import Model
from detector.utils.torch_utils import select_device

# 方式一：加载训练好的模型
device = select_device('0')
model = attempt_load('best.pt', map_location=device)
model.eval()

# 方式二：从配置文件创建新模型
model = Model('cfg/yolov7-tiny.yaml', ch=3, nc=80)

# 方式三：加载预训练权重到新模型
ckpt = torch.load('yolov7-tiny.pt', map_location=device)
model = Model('cfg/yolov7-tiny.yaml', ch=3, nc=80)
model.load_state_dict(ckpt['model'].state_dict(), strict=False)
```

### 单张图像推理

```python
import cv2
import torch
import numpy as np
from detector.models.experimental import attempt_load
from detector.utils.general import non_max_suppression, scale_coords, letterbox
from detector.utils.torch_utils import select_device

def predict_single_image(image_path, weights_path, conf_thres=0.25, iou_thres=0.45, img_size=640):
    """
    单张图像推理
    
    Args:
        image_path: 图像路径
        weights_path: 权重路径
        conf_thres: 置信度阈值
        iou_thres: IoU 阈值
        img_size: 推理尺寸
    
    Returns:
        detections: 检测结果列表 [(x1, y1, x2, y2, conf, cls), ...]
    """
    # 加载模型
    device = select_device('0')
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    
    # 读取图像
    img0 = cv2.imread(image_path)
    h, w = img0.shape[:2]
    
    # 预处理
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)  # add batch dimension
    
    # 推理
    with torch.no_grad():
        pred = model(img)[0]
    
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # 处理结果
    detections = []
    for det in pred:
        if len(det):
            # 将坐标缩放到原图尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                detections.append((*xyxy, conf.item(), int(cls.item())))
    
    return detections

# 使用示例
detections = predict_single_image('test.jpg', 'best.pt')
for x1, y1, x2, y2, conf, cls in detections:
    print(f"Class: {cls}, Conf: {conf:.2f}, BBox: [{x1}, {y1}, {x2}, {y2}]")
```

### 批量图像推理

```python
import torch
from detector.models.experimental import attempt_load
from detector.utils.datasets import LoadImages
from detector.utils.general import non_max_suppression, scale_coords
from detector.utils.torch_utils import select_device, time_synchronized

def predict_batch(source_path, weights_path, conf_thres=0.25, iou_thres=0.45, img_size=640):
    """
    批量图像推理
    
    Args:
        source_path: 图像目录或文件路径
        weights_path: 权重路径
        conf_thres: 置信度阈值
        iou_thres: IoU 阈值
        img_size: 推理尺寸
    
    Yields:
        path: 图像路径
        det: 检测结果
        img: 原始图像
    """
    # 加载模型
    device = select_device('0')
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    
    # 加载数据
    dataset = LoadImages(source_path, img_size=img_size)
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # 推理
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img)[0]
        t2 = time_synchronized()
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t3 = time_synchronized()
        
        # 处理结果
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        
        print(f'Inference: {(t2-t1)*1000:.1f}ms, NMS: {(t3-t2)*1000:.1f}ms')
        yield path, det, im0s

# 使用示例
for path, det, img in predict_batch('./images', 'best.pt'):
    print(f"{path}: {len(det)} detections")
    for *xyxy, conf, cls in det:
        print(f"  Class {int(cls)}: {conf:.2f}")
```

---

## 完整示例

### 完整训练流程

```python
"""
完整的模型训练示例
"""
import os
import yaml
import torch
from pathlib import Path
from detector.train import train
from detector.test import test
from detector.utils.torch_utils import select_device
from detector.utils.general import increment_path

def train_model(
    data_yaml='data/data.yaml',
    cfg_yaml='cfg/yolov7-tiny.yaml',
    hyp_yaml='data/hyp.scratch.tiny.yaml',
    weights='yolov7-tiny.pt',
    epochs=100,
    batch_size=16,
    img_size=640,
    device='0',
    project='runs/train',
    name='exp'
):
    """完整训练流程"""
    
    # 加载超参数
    with open(hyp_yaml) as f:
        hyp = yaml.safe_load(f)
    
    # 创建配置对象
    class Opt:
        def __init__(self):
            self.weights = weights
            self.cfg = cfg_yaml
            self.data = data_yaml
            self.epochs = epochs
            self.batch_size = batch_size
            self.img_size = [img_size, img_size]
            self.device = device
            self.workers = 8
            self.project = project
            self.name = name
            self.exist_ok = False
            self.rect = False
            self.resume = False
            self.nosave = False
            self.notest = False
            self.noautoanchor = False
            self.evolve = False
            self.cache_images = False
            self.image_weights = False
            self.multi_scale = False
            self.single_cls = False
            self.adam = False
            self.sync_bn = False
            self.local_rank = -1
            self.world_size = 1
            self.global_rank = -1
            self.freeze = [0]
            self.quad = False
            self.linear_lr = False
            self.label_smoothing = 0.0
            self.save_period = -1
            self.v5_metric = False
            self.save_dir = str(increment_path(Path(project) / name, exist_ok=False))
            self.total_batch_size = batch_size
    
    opt = Opt()
    device = select_device(device)
    
    # 开始训练
    print(f"Training started, results will be saved to: {opt.save_dir}")
    results = train(hyp, opt, device)
    
    return results, opt.save_dir

# 运行训练
if __name__ == '__main__':
    results, save_dir = train_model(
        data_yaml='data/data.yaml',
        cfg_yaml='cfg/yolov7-tiny.yaml',
        weights='yolov7-tiny.pt',
        epochs=100,
        batch_size=16,
        device='0'
    )
    
    print(f"\nTraining completed!")
    print(f"mAP@0.5: {results[2]:.4f}")
    print(f"mAP@0.5:0.95: {results[3]:.4f}")
    print(f"Best weights: {save_dir}/weights/best.pt")
```

### 完整推理流程

```python
"""
完整的模型推理示例
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from detector.models.experimental import attempt_load
from detector.utils.general import non_max_suppression, scale_coords, letterbox, plot_one_box
from detector.utils.torch_utils import select_device
from detector.utils.plots import colors

class Detector:
    """目标检测器类"""
    
    def __init__(self, weights_path, device='0', img_size=640, conf_thres=0.25, iou_thres=0.45):
        """
        初始化检测器
        
        Args:
            weights_path: 权重文件路径
            device: 推理设备
            img_size: 推理图像尺寸
            conf_thres: 置信度阈值
            iou_thres: IoU 阈值
        """
        self.device = select_device(device)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 加载模型
        self.model = attempt_load(weights_path, map_location=self.device)
        self.model.eval()
        
        # 获取类别名称
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
    
    def preprocess(self, img0):
        """图像预处理"""
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
    
    def detect(self, image):
        """
        检测单张图像
        
        Args:
            image: BGR 图像数组或图像路径
        
        Returns:
            detections: 检测结果列表
            img0: 原始图像
        """
        # 读取图像
        if isinstance(image, str):
            img0 = cv2.imread(image)
        else:
            img0 = image.copy()
        
        # 预处理
        img = self.preprocess(img0)
        
        # 推理
        with torch.no_grad():
            pred = self.model(img)[0]
        
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        # 处理结果
        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:
                    detections.append({
                        'bbox': [int(x.item()) for x in xyxy],
                        'confidence': conf.item(),
                        'class_id': int(cls.item()),
                        'class_name': self.names[int(cls.item())]
                    })
        
        return detections, img0
    
    def draw_detections(self, img0, detections, output_path=None):
        """绘制检测结果"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = colors(det['class_id'], True)
            cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if output_path:
            cv2.imwrite(output_path, img0)
        
        return img0
    
    def detect_and_save(self, image_path, output_dir):
        """检测并保存结果"""
        detections, img0 = self.detect(image_path)
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        self.draw_detections(img0, detections, output_path)
        
        return detections

# 使用示例
if __name__ == '__main__':
    # 初始化检测器
    detector = Detector(
        weights_path='runs/train/exp/weights/best.pt',
        device='0',
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45
    )
    
    # 单张图像检测
    detections, img = detector.detect('test.jpg')
    for det in detections:
        print(f"{det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
    
    # 绘制并保存结果
    detector.draw_detections(img, detections, 'result.jpg')
    
    # 批量检测
    image_dir = './images'
    output_dir = './results'
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_dir, img_name)
            detector.detect_and_save(img_path, output_dir)
```

---

## 示例代码

完整的推理示例代码位于 [`examples/inference_example.py`](examples/inference_example.py)，包含：

- 从 PT 权重加载模型
- 打印模型各层名称
- 单张图像推理
- 检测结果可视化

运行示例：

```bash
# 基本推理
python examples/inference_example.py --weights best.pt --image test.jpg

# 打印模型层信息
python examples/inference_example.py --weights best.pt --image test.jpg --print-layers

# 指定参数
python examples/inference_example.py --weights best.pt --image test.jpg \
    --img-size 1024 --conf-thres 0.3 --iou-thres 0.5 --device 0
```

---

## 相关文件

- [`detector/__init__.py`](detector/__init__.py) - 包入口，导出核心函数
- [`detector/train.py`](detector/train.py) - 训练主逻辑
- [`detector/test.py`](detector/test.py) - 测试/评估主逻辑
- [`detector/predict_bbox.py`](detector/predict_bbox.py) - 批量推理逻辑
- [`detector/show_detector_results.py`](detector/show_detector_results.py) - 结果可视化
- [`detector/models/yolo.py`](detector/models/yolo.py) - 模型定义
- [`detector/models/experimental.py`](detector/models/experimental.py) - 实验性模型功能
- [`detector/utils/datasets.py`](detector/utils/datasets.py) - 数据加载工具
- [`detector/utils/general.py`](detector/utils/general.py) - 通用工具函数
- [`detector/utils/torch_utils.py`](detector/utils/torch_utils.py) - PyTorch 工具函数

---

## 更多信息

- [CLI 接口文档](CLI_API.md) - 命令行接口使用说明
- [架构文档](ARCHITECTURE.md) - 项目架构设计说明
- [README](README.md) - 项目概述和安装说明