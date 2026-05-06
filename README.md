# Detector Train - 目标检测模型训练框架

目标检测模型训练和推理框架，支持多 GPU 分布式训练、混合精度训练、模型导出ONNX、模型推理等功能。当前版本基于YOLO7进行微调修改而得到。

## 安装

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/example/detector-train.git
cd detector-train

# 安装为 Python 包（开发模式）
pip install -e .

# 或者安装为普通包
pip install .
```

### 依赖要求

- Python >= 3.7
- PyTorch >= 1.7.0
- CUDA >= 10.2 (可选，用于 GPU 加速)

主要依赖会在安装时自动安装。

## 使用方法

## 命令行工具完整列表

| 命令 | 说明 |
|------|------|
| `detector-train` | 训练目标检测模型 |
| `detector-test` | 测试/验证模型 |
| `detector-detect` | 使用模型进行单张图像目标检测 |
| `detector-predict` | 批量推理 - 对图像目录进行批量检测，输出 YOLO 格式结果 |
| `export-onnx` | 导出模型为 ONNX 格式 |
| `export-fake-rgbt` | 导出 Fake RGBT 模型 |
| `export-scale-conf` | 导出尺度置信度图 |
| `detector-trim` | 裁剪模型（移除训练模块） |
| `detector-rename` | 重命名模型类别 |
| `compute-anchors` | 计算数据集锚框 |
| `print-anchors` | 打印锚框信息 |
| `detector-analyze` | 分析训练结果 |
| `detector-generate-templates` | 生成模板配置文件 |


### 命令行方式

安装完成后，可以使用以下命令：

#### 训练

```bash
# 基本训练命令
detector-train \
    --weights ckpoints/opencar.pt \
    --cfg cfg/yolov7-tiny-silu-sqnet.yaml \
    --data data/data_test.yaml \
    --hyp data/hyp.scratch.tiny.yaml \
    --epochs 20 \
    --batch-size 32 \
    --img-size 1024 1024 \
    --device 0
```

#### 测试/验证

```bash
# 基本测试命令
detector-test \
    --weights runs/train/exp/weights/best.pt \
    --data data/data_test.yaml \
    --batch-size 32 \
    --img-size 1024 \
    --device 0 \
    --task val

# 测试模式（test）
detector-test \
    --weights runs/train/exp/weights/best.pt \
    --data data/data_test.yaml \
    --task test

# 速度基准测试
detector-test \
    --weights runs/train/exp/weights/best.pt \
    --data data/data_test.yaml \
    --task speed
```

#### 目标检测

```bash
# 使用模型进行目标检测
detector-detect \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/image.jpg \
    --img-size 1024 \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --device 0 \
    --save-txt \
    --save-conf
```

#### 批量推理

对应原来的 testx2rgb.py

```bash
# 单文件夹模式 - 对单个图像文件夹进行批量检测
detector-predict \
    --weights runs/train/exp/weights/best.pt \
    --voc /path/to/images \
    --batch-size 32 \
    --img-size 1024 \
    --conf-thres 0.01 \
    --iou-thres 0.45

# 多文件夹模式 - 对多个图像文件夹进行批量检测
detector-predict \
    --weights runs/train/exp/weights/best.pt \
    --home /path/to/data_home \
    --batch-size 32 \
    --img-size 1024
```

#### 模型导出

```bash
# 导出 ONNX 模型
export-onnx \
    --weights runs/train/exp/weights/best.pt \
    --img-size 1024 1024 \
    --batch-size 1 \
    --device cpu \
    --simplify

# 导出 Fake RGBT 模型（用于双光模型转换）
export-fake-rgbt \
    --weights runs/train/exp/weights/best.pt \
    --output runs/export/rgbt_model.pt
```

#### 模型工具

```bash
# 裁剪模型（移除训练相关模块）
detector-trim \
    --weights runs/train/exp/weights/best.pt \
    --output runs/export/trimmed_model.pt

# 重命名模型中的类别
detector-rename \
    --weights runs/train/exp/weights/best.pt \
    --new-names cfg/new_classnames.txt \
    --output runs/export/renamed_model.pt

# 计算数据集的锚框
compute-anchors \
    --data data/data_test.yaml \
    --anchors 9 \
    --img-size 1024

# 打印锚框信息
print-anchors \
    --anchors "[10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]"
```

#### 分析工具

```bash
# 分析训练结果
detector-analyze \
    --results-dir runs/train \
    --plot-loss \
    --plot-metrics
```

#### 模板生成

```bash
# 生成模板配置文件到指定目录
detector-generate-templates \
    --output ./my_project

# 覆盖已存在的文件
detector-generate-templates \
    --output ./my_project \
    --force
```

生成的文件：
- `data.yaml` - 数据集配置文件
- `cfg.yaml` - 模型配置文件
- `hyp.yaml` - 超参数配置文件

使用生成的模板：
```bash
detector-train --data ./my_project/data.yaml --cfg ./my_project/cfg.yaml --hyp ./my_project/hyp.yaml
```

### Python API 方式

也可以直接在 Python 代码中导入并使用：

```python
from detector import train, test
import argparse

# 训练
opt = argparse.Namespace(
    weights='ckpoints/opencar.pt',
    cfg='cfg/yolov7-tiny-silu-sqnet.yaml',
    data='data/data_test.yaml',
    hyp='data/hyp.scratch.tiny.yaml',
    epochs=20,
    batch_size=32,
    img_size=[1024, 1024],
    device='0',
    project='runs/train',
    name='exp',
    exist_ok=False,
    # ... 其他参数
)

# 训练完成后，可以进行测试
test(
    data='data/data_test.yaml',
    weights='runs/train/exp/weights/best.pt',
    batch_size=32,
    imgsz=1024,
    device='0'
)
```

## 主要参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `ckpoints/opencar.pt` | 初始权重路径 |
| `--cfg` | `cfg/yolov7-tiny-silu-sqnet.yaml` | 模型配置文件 |
| `--data` | `data/data_test.yaml` | 数据集配置文件 |
| `--hyp` | `data/hyp.scratch.tiny.yaml` | 超参数配置文件 |
| `--epochs` | `20` | 训练轮数 |
| `--batch-size` | `32` | 批次大小 |
| `--img-size` | `[1024, 1024]` | 图像尺寸 [train, test] |
| `--device` | `''` | CUDA 设备，如 '0' 或 '0,1,2,3' |
| `--resume` | `False` | 恢复训练 |
| `--rect` | `False` | 矩形训练 |
| `--sync-bn` | `False` | 使用 SyncBatchNorm |
| `--adam` | `False` | 使用 Adam 优化器 |

### 测试参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `yolov7.pt` | 模型权重路径 |
| `--data` | `data/coco.yaml` | 数据集配置文件 |
| `--batch-size` | `32` | 批次大小 |
| `--img-size` | `1024` | 推理图像尺寸 |
| `--conf-thres` | `0.001` | 置信度阈值 |
| `--iou-thres` | `0.45` | NMS 的 IOU 阈值 |
| `--task` | `val` | 任务类型：train/val/test/speed/study |
| `--device` | `''` | CUDA 设备 |
| `--save-json` | `False` | 保存 COCO JSON 结果 |
| `--verbose` | `False` | 按类别报告 mAP |

## 命令行工具完整列表

| 命令 | 说明 |
|------|------|
| `detector-train` | 训练目标检测模型 |
| `detector-test` | 测试/验证模型 |
| `detector-detect` | 使用模型进行单张图像目标检测 |
| `detector-predict` | 批量推理 - 对图像目录进行批量检测，输出 YOLO 格式结果 |
| `export-onnx` | 导出模型为 ONNX 格式 |
| `export-fake-rgbt` | 导出 Fake RGBT 模型 |
| `export-scale-conf` | 导出尺度置信度图 |
| `detector-trim` | 裁剪模型（移除训练模块） |
| `detector-rename` | 重命名模型类别 |
| `compute-anchors` | 计算数据集锚框 |
| `print-anchors` | 打印锚框信息 |
| `detector-analyze` | 分析训练结果 |
| `detector-generate-templates` | 生成模板配置文件 |

## 项目结构

```
detector-train/
├── detector/              # 主包
│   ├── __init__.py
│   ├── train.py           # 训练核心模块
│   ├── test.py            # 测试核心模块
│   ├── predict_bbox.py    # 预测框处理
│   ├── show_detector_results.py  # 结果可视化
│   ├── cli/               # 命令行接口
│   │   ├── __init__.py
│   │   ├── train_cli.py   # 训练 CLI
│   │   ├── test_cli.py    # 测试 CLI
│   │   ├── detect.py      # 检测 CLI
│   │   ├── export_onnx.py # ONNX 导出 CLI
│   │   ├── export_fake_rgbt.py
│   │   ├── export_scale_conf.py
│   │   ├── model_trim.py  # 模型裁剪 CLI
│   │   ├── model_rename_classes.py
│   │   ├── compute_anchors.py
│   │   ├── print_anchors.py
│   │   └── analyze_list.py
│   ├── models/            # 模型定义
│   │   ├── __init__.py
│   │   ├── common.py      # 通用模块
│   │   ├── experimental.py # 实验性模块
│   │   └── yolo.py        # YOLO 模型
│   └── utils/             # 工具函数
│       ├── __init__.py
│       ├── activations.py
│       ├── autoanchor.py
│       ├── datasets.py
│       ├── general.py
│       ├── google_utils.py
│       ├── loss.py
│       ├── metrics.py
│       ├── plots.py
│       ├── tools.py
│       ├── torch_utils.py
│       ├── aws/
│       ├── google_app_engine/
│       └── wandb_logging/
├── tools/                 # 工具脚本
│   ├── get_blocked_map.py
│   ├── get_cam_map.py
│   ├── loss_landscapev5.py
│   └── tsne-yolov7-tiny_forbboxv5.py
├── cfg/                   # 模型配置
│   ├── yolov7-tiny-silu-sqnet.yaml
│   └── yolov7-tiny-silu.yaml
├── data/                  # 数据配置
│   ├── data_test.yaml
│   ├── hyp.finetune.tiny.yaml
│   └── hyp.scratch.tiny.yaml
├── ckpoints/              # 预训练权重
├── scripts/               # Shell 脚本
├── setup.py               # 安装脚本
├── ARCHITECTURE.md        # 架构文档
└── README.md              # 本文档
```

## 数据集配置

数据集配置文件采用 YAML 格式，示例如下：

```yaml
# data/data_test.yaml
train: /path/to/train/images
val: /path/to/val/images
test: /path/to/test/images

nc: 1  # 类别数量
names: ['object']  # 类别名称
```

## TODO

下一步计划，在该框架下吸收更新，更先进的模型经验。

## 许可证

MIT License

## 致谢

本项目基于 YOLOv7 开发，感谢原作者的贡献。
