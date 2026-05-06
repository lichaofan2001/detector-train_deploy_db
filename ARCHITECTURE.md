# 目标检测模型代码架构分析

## 一、项目概述

本项目是一个基于 YOLOv7 的目标检测模型训练和推理框架，支持多 GPU 分布式训练、混合精度训练、EMA（指数移动平均）、W&B 日志记录等功能。

## 二、目录结构

```
detector-train/
├── setup.py                 # 安装脚本
├── detector/                # 主程序包
│   ├── __init__.py
│   ├── train.py             # 训练主脚本
│   ├── test.py              # 测试/验证主脚本
│   ├── predict_bbox.py      # 预测脚本
│   ├── show_detector_results.py  # 结果展示脚本
│   ├── cli/                 # 命令行接口
│   │   ├── __init__.py
│   │   ├── train_cli.py     # 训练 CLI
│   │   ├── test_cli.py      # 测试 CLI
│   │   ├── detect.py        # 推理 CLI
│   │   ├── export_onnx.py   # ONNX 导出 CLI
│   │   ├── model_trim.py    # 模型裁剪 CLI
│   │   └── ...
│   ├── models/              # 模型定义
│   │   ├── __init__.py
│   │   ├── common.py        # 通用模块定义
│   │   ├── experimental.py  # 实验性模块
│   │   └── yolo.py          # YOLO 模型定义
│   └── utils/               # 工具函数
│       ├── __init__.py
│       ├── activations.py   # 激活函数
│       ├── autoanchor.py    # 自动锚框
│       ├── datasets.py      # 数据集加载
│       ├── general.py       # 通用工具
│       ├── loss.py          # 损失函数
│       ├── metrics.py       # 评估指标
│       ├── plots.py         # 可视化
│       ├── torch_utils.py   # PyTorch 工具
│       ├── aws/             # AWS 相关工具
│       ├── google_app_engine/
│       └── wandb_logging/   # W&B 日志工具
├── tools/                   # 工具脚本
│   ├── get_blocked_map.py   # 遮挡图生成
│   ├── get_cam_map.py       # CAM 热力图生成
│   ├── loss_landscapev5.py  # 损失景观可视化
│   └── tsne-yolov7-tiny_forbboxv5.py  # t-SNE 可视化
├── cfg/                     # 模型配置文件
│   ├── yolov7-tiny-silu-sqnet.yaml
│   └── yolov7-tiny-silu.yaml
├── data/                    # 数据配置文件
│   ├── data_test.yaml       # 测试数据集配置
│   ├── hyp.finetune.tiny.yaml  # 微调超参数
│   └── hyp.scratch.tiny.yaml   # 从头训练超参数
├── ckpoints/                # 预训练权重目录
├── scripts/                 # Shell 脚本
└── ARCHITECTURE.md          # 架构文档
```

## 三、核心模块分析

### 3.1 训练模块 (train.py)

**主要功能：**
- 支持单卡/多卡分布式训练（DDP）
- 混合精度训练（AMP）
- EMA 模型更新
- 自动锚框检查
- W&B 和 TensorBoard 日志记录
- 学习率调度（Cosine/Linear）
- 模型断点续训

**核心函数：**
- `train(hyp, opt, device, tb_writer)`: 主训练循环
- `close_mosaic(dataloader)`: 关闭 Mosaic 数据增强

**命令行参数：**
- `--weights`: 初始权重路径
- `--cfg`: 模型配置文件
- `--data`: 数据集配置文件
- `--hyp`: 超参数配置文件
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--img-size`: 图像尺寸
- `--device`: CUDA 设备
- 等等...

### 3.2 测试模块 (test.py)

**主要功能：**
- 验证集/测试集评估
- mAP 计算（COCO 标准和 YOLOv5 标准）
- 混淆矩阵生成
- 模型导出支持（ONNX Trace）
- 速度基准测试

**核心函数：**
- `test(data, weights, batch_size, imgsz, ...)`: 主测试函数

**命令行参数：**
- `--weights`: 模型权重路径
- `--data`: 数据集配置文件
- `--batch-size`: 批次大小
- `--img-size`: 图像尺寸
- `--conf-thres`: 置信度阈值
- `--iou-thres`: NMS IOU 阈值
- `--task`: 任务类型（train/val/test/speed/study）
- 等等...

### 3.3 模型模块 (detector/models/)

**common.py:**
- 包含各种卷积模块（Conv, Bottleneck, SPP 等）
- 注意力机制模块
- YOLOv7 特有的模块定义

**yolo.py:**
- `Model` 类：YOLO 模型主类
- 模型构建、前向传播、损失计算
- 锚框处理

**experimental.py:**
- 实验性模块和工具函数
- 模型导出相关功能

### 3.4 工具模块 (detector/utils/)

**datasets.py:**
- `create_dataloader()`: 创建数据加载器
- `LoadImagesAndLabels`: 图像和标签加载
- Mosaic、MixUp 等数据增强

**loss.py:**
- `ComputeLoss`: YOLO 损失计算
- `ComputeLossOTA`: OTA 标签分配损失

**metrics.py:**
- `ap_per_class()`: 各类别 AP 计算
- `ConfusionMatrix`: 混淆矩阵

**general.py:**
- 通用工具函数
- 文件操作、路径处理
- NMS 后处理

**torch_utils.py:**
- `select_device()`: 设备选择
- `ModelEMA`: EMA 实现
- 分布式训练工具

## 四、数据流

### 4.1 训练流程
```
1. 解析命令行参数 → 2. 初始化设备/日志
        ↓
3. 加载模型（预训练/从头开始）
        ↓
4. 创建数据加载器（train/val）
        ↓
5. 初始化优化器/调度器/EMA
        ↓
6. 训练循环（epoch）
   ├── 前向传播
   ├── 损失计算
   ├── 反向传播
   └── 优化器更新
        ↓
7. 验证评估
        ↓
8. 保存检查点
        ↓
9. 重复 6-8 直到完成
```

### 4.2 测试流程
```
1. 解析命令行参数 → 2. 加载模型
        ↓
3. 创建数据加载器
        ↓
4. 推理循环
   ├── 前向传播
   └── NMS 后处理
        ↓
5. 指标计算（mAP, 混淆矩阵等）
        ↓
6. 结果保存/可视化
```

## 五、重构计划

### 5.1 命令行接口重构
将 `train.py` 和 `test.py` 中的核心逻辑提取为 Python 包的可调用函数，支持：
- 通过 Python API 直接调用
- 通过命令行工具调用

### 5.2 包结构设计
```
detector_train/
├── __init__.py          # 包初始化
├── cli/                 # 命令行接口
│   ├── __init__.py
│   ├── train.py         # 训练 CLI
│   └── test.py          # 测试 CLI
├── train.py             # 训练核心逻辑（重构后）
├── test.py              # 测试核心逻辑（重构后）
├── models/              # 模型模块（复用）
└── utils/               # 工具模块（复用）
```

### 5.3 setup.py 配置
- 定义包元数据
- 配置入口点（entry_points）
- 管理依赖关系

## 六、命令行接口

本项目提供了多个命令行工具，用于模型训练、测试、推理和工具操作。

### 6.1 训练与测试

#### detector-train - 模型训练
用于训练 YOLOv7 目标检测模型。

**主要功能：**
- 支持单卡/多卡分布式训练（DDP）
- 混合精度训练（AMP）
- EMA 模型更新
- 自动锚框检查
- W&B 和 TensorBoard 日志记录

**常用参数：**
- `--weights`: 初始权重文件路径
- `--cfg`: 模型配置文件路径 (model.yaml)
- `--data`: 数据集配置文件路径 (data.yaml)
- `--hyp`: 超参数配置文件路径
- `--epochs`: 训练轮数
- `--batch-size`: 所有 GPU 的总批次大小
- `--img-size`: 训练和测试图像尺寸
- `--device`: CUDA 设备（如 0 或 0,1,2,3 或 cpu）
- `--resume`: 恢复最近一次训练

**示例：**
```bash
detector-train --weights ckpoints/yolov7-tiny.pt --cfg cfg/yolov7-tiny-silu-sqnet.yaml --data data/data_test.yaml --epochs 100 --batch-size 32
```

#### detector-test - 模型测试/验证
用于评估模型性能，计算 mAP 等指标。

**主要功能：**
- 验证集/测试集评估
- mAP 计算（COCO 标准和 YOLOv5 标准）
- 混淆矩阵生成
- 速度基准测试

**常用参数：**
- `--weights`: 模型权重文件路径
- `--data`: 数据集配置文件路径
- `--batch-size`: 每个图像批次的大小
- `--img-size`: 推理图像尺寸
- `--conf-thres`: 目标置信度阈值
- `--iou-thres`: NMS 的 IOU 阈值
- `--task`: 任务类型（train/val/test/speed/study）

**示例：**
```bash
detector-test --weights runs/train/exp/weights/best.pt --data data/data_test.yaml --batch-size 32 --img-size 1024
```

### 6.2 推理

#### detector-detect - 目标检测推理
用于在图像、视频或摄像头流上运行目标检测。

**主要功能：**
- 支持图像、视频、摄像头输入
- 实时检测结果显示
- 检测结果保存（图像/文本）

**常用参数：**
- `--weights`: 模型权重文件路径
- `--source`: 检测源（文件/文件夹，0 表示摄像头）
- `--img-size`: 推理图像尺寸
- `--conf-thres`: 目标置信度阈值
- `--iou-thres`: NMS 的 IOU 阈值
- `--save-txt`: 将结果保存到 *.txt 文件
- `--view-img`: 显示检测结果

**示例：**
```bash
detector-detect --weights yolov7.pt --source inference/images --img-size 1024 --conf-thres 0.15
```

### 6.3 模型导出

#### export-onnx - 导出 ONNX 模型
用于将 PyTorch 模型导出为 ONNX 格式。

**主要功能：**
- 支持动态轴和动态批次大小
- 可选 NMS 集成
- 支持 TensorRT 和 ONNX Runtime

**常用参数：**
- `--weights`: 模型权重文件路径（必填）
- `--img-size`: 图像尺寸（高，宽）
- `--batch-size`: 批次大小
- `--dynamic`: 动态 ONNX 轴
- `--simplify`: 简化 ONNX 模型
- `--include-nms`: 导出带 NMS 的端到端 ONNX
- `--opset`: ONNX opset 版本

**示例：**
```bash
export-onnx --weights model.pt --img-size 640 640 --simplify --include-nms
```

#### export-fake-rgbt - 导出 Fake RGBT ONNX 模型
用于导出带有假 RGBT 输入（RGB + IR 通道）的 ONNX 模型。

**常用参数：**
- `--weights`: 模型权重文件路径（必填）
- `--img-size`: 图像尺寸
- `--batch-size`: 批次大小
- `--dynamic`: 动态 ONNX 轴
- `--simplify`: 简化 ONNX 模型
- `--include-nms`: 导出带 NMS 的端到端 ONNX

**示例：**
```bash
export-fake-rgbt --weights model.pt --img-size 640 640 --simplify
```

#### export-scale-conf - 导出带缩放置信度的 ONNX 模型
用于导出带有缩放置信度值的 ONNX 模型。

**常用参数：**
- `--weights`: 模型权重文件路径（必填）
- `--img-size`: 图像尺寸
- `--scale-conf`: 置信度缩放因子
- `--end2end`: 导出端到端 ONNX
- `--simplify`: 简化 ONNX 模型
- `--include-nms`: 导出带 NMS 的端到端 ONNX

**示例：**
```bash
export-scale-conf --weights model.pt --img-size 1024 1024 --scale-conf 0.5 --include-nms
```

### 6.4 模型工具

#### detector-trim - 模型裁剪
用于从训练好的模型中裁剪指定类别。

**主要功能：**
- 裁剪不需要的类别
- 减少模型输出维度

**常用参数：**
- `--weights`: 模型权重文件路径（必填）
- `--to-trim`: 要裁剪的类别索引，逗号分隔（必填）
- `--out-file`: 输出文件路径
- `--force`: 无需确认直接覆盖

**示例：**
```bash
detector-trim --weights model.pt --to-trim "3,4" --out-file model_trimmed.pt
```

#### detector-rename - 重命名模型类别
用于重命名训练好的模型中的类别名称。

**主要功能：**
- 更新模型中的类别名称
- 保持模型权重不变

**常用参数：**
- `--weights`: 模型权重文件路径（必填）
- `--newnames`: 新的类别名称，逗号分隔（必填）
- `--out-file`: 输出文件路径

**示例：**
```bash
detector-rename --weights model.pt --newnames "car,person,dog" --out-file model_renamed.pt
```

#### compute-anchors - 计算最优锚框
使用 k-means 聚类计算数据集的最优锚框。

**主要功能：**
- 基于数据集标注计算最优锚框
- 使用 k-means 聚类算法

**常用参数：**
- `--data`: 数据集配置 YAML 文件路径（必填）
- `--num-anchors`: 要计算的锚框数量（默认：9）
- `--img-size`: 锚框计算的图像尺寸（默认：1024）
- `--thresh`: k-means 的 IoU 阈值（默认：5.0）
- `--iters`: k-means 迭代次数（默认：5000）

**示例：**
```bash
compute-anchors --data data/dataset.yaml --num-anchors 9 --img-size 1024
```

#### print-anchors - 打印锚框
用于打印模型中的锚框。

**常用参数：**
- `--weights`: 模型权重文件路径（必填）

**示例：**
```bash
print-anchors --weights model.pt
```

### 6.5 分析工具

#### detector-analyze - 分析检测结果
用于分析检测结果中的假阳性和假阴性。

**主要功能：**
- 分析误检和漏检
- 可视化检测结果
- 生成错误列表

**常用参数：**
- `--weights`: 模型权重文件路径
- `--data`: 数据列表 txt 文件路径（必填）
- `--img-size`: 推理图像尺寸
- `--conf-thres`: 目标置信度阈值
- `--iou-thres`: NMS 的 IOU 阈值
- `--save-dir`: 保存结果的目录
- `--no-draw`: 不绘制图像
- `--error-list`: 保存错误列表文件的路径

**示例：**
```bash
detector-analyze --data data/test_list.txt --weights model.pt --save-dir results/analysis
```

## 七、依赖关系

主要依赖：
- torch >= 1.7.0
- torchvision
- numpy
- opencv-python
- pyyaml
- tqdm
- tensorboard
- wandb (可选)
- pycocotools (可选，用于 COCO 评估)
