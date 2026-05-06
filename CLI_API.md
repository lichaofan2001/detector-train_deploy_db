# Detector CLI 接口文档

本文档详细描述了 `detector` 包提供的所有命令行接口（CLI），包括每个命令的功能、参数说明以及调用示例。

## 目录

- [安装与配置](#安装与配置)
- [命令列表](#命令列表)
  - [detector-train](#detector-train) - 模型训练
  - [detector-test](#detector-test) - 模型测试/评估
  - [detector-detect](#detector-detect) - 目标检测推理
  - [detector-show](#detector-show) - 显示检测结果
  - [detector-predict](#detector-predict) - 批量推理
  - [compute-anchors](#compute-anchors) - 计算锚框
  - [print-anchors](#print-anchors) - 打印锚框
  - [detector-trim](#detector-trim) - 裁剪模型类别
  - [detector-rename](#detector-rename) - 重命名模型类别
  - [export-onnx](#export-onnx) - 导出 ONNX 模型
  - [export-fake-rgbt](#export-fake-rgbt) - 导出 Fake RGBT ONNX 模型
  - [export-scale-conf](#export-scale-conf) - 导出带缩放置信度的 ONNX 模型
  - [detector-analyze](#detector-analyze) - 分析检测结果
  - [detector-generate-templates](#detector-generate-templates) - 生成模板配置文件

---

## 安装与配置

在使用以下命令之前，请先安装 `detector` 包：

```bash
# 开发模式安装
pip install -e .

# 或作为普通包安装
pip install .
```

### 预训练权重

安装包会自动包含 `ckpoints` 目录中的预训练权重文件。默认情况下，训练和测试命令使用 `~/.detector/ckpoints/opencar.pt` 作为初始权重。


---

## 命令列表

---

### detector-train

**功能**: 训练目标检测模型。支持 DDP 分布式训练、超参数进化、TensorBoard 日志记录等功能。

**命令**: `detector-train`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | `'ckpoints/yolov7-tiny.pt'` | 初始权重文件路径 |
| `--cfg` | str | `'cfg/yolov7-tiny-silu-sqnet.yaml'` | 模型配置文件路径 (model.yaml) |
| `--data` | str | `'data/data_test.yaml'` | 数据集配置文件路径 (data.yaml) |
| `--hyp` | str | `'data/hyp.scratch.tiny.yaml'` | 超参数配置文件路径 |
| `--epochs` | int | `20` | 训练轮数 (epochs) |
| `--batch-size` | int | `32` | 所有 GPU 的总批次大小 (total batch size) |
| `--img-size` | list | `[1024, 1024]` | 训练和测试图像尺寸 [train, test] |
| `--rect` | bool | `False` | 矩形训练 (rectangular training) |
| `--resume` | str/nargs | `False` | 恢复最近一次训练 |
| `--nosave` | bool | `False` | 仅保存最终检查点 |
| `--notest` | bool | `False` | 仅测试最终 epoch |
| `--noautoanchor` | bool | `False` | 禁用 autoanchor 检查 |
| `--evolve` | bool | `False` | 进化超参数 |
| `--bucket` | str | `''` | gsutil 存储桶 |
| `--cache-images` | bool | `False` | 缓存图像以加速训练 |
| `--image-weights` | bool | `False` | 使用加权图像选择进行训练 |
| `--device` | str | `''` | CUDA 设备，如 `0` 或 `0,1,2,3` 或 `cpu` |
| `--multi-scale` | bool | `False` | 变化图像尺寸 +/- 50% |
| `--single-cls` | bool | `False` | 将多类别数据作为单类别训练 |
| `--adam` | bool | `False` | 使用 torch.optim.Adam() 优化器 |
| `--sync-bn` | bool | `False` | 使用 SyncBatchNorm，仅在 DDP 模式下可用 |
| `--local_rank` | int | `-1` | DDP 参数，请勿修改 |
| `--workers` | int | `16` | 数据加载器最大工作线程数 |
| `--project` | str | `'runs/train'` | 保存结果到 project/name |
| `--entity` | str | `None` | W&B 实体名称 |
| `--name` | str | `'exp'` | 保存结果到 project/name |
| `--exist-ok` | bool | `False` | 已存在的 project/name 不递增 |
| `--quad` | bool | `False` | 使用四路数据加载器 |
| `--linear-lr` | bool | `False` | 线性学习率 |
| `--label-smoothing` | float | `0.05` | 标签平滑 epsilon |
| `--upload_dataset` | bool | `False` | 将数据集作为 W&B artifact 上传 |
| `--bbox_interval` | int | `-1` | 设置 W&B 边界框图像日志间隔 |
| `--save_period` | int | `-1` | 每 save_period 个 epoch 保存一次模型 |
| `--artifact_alias` | str | `'latest'` | 要使用的数据集 artifact 版本 |
| `--freeze` | list | `[0]` | 冻结层：yolov7 主干=50, 前 3 层=0 1 2 |
| `--v5-metric` | bool | `False` | 在 AP 计算中假设最大召回率为 1.0 |

**调用示例**:

```bash
# 基本训练命令
detector-train --data data/data.yaml --cfg cfg/yolov7-tiny-silu.yaml --weights '' --batch-size 16 --epochs 300

# 从预训练权重开始训练
detector-train --data data/data.yaml --cfg cfg/yolov7-tiny-silu.yaml --weights yolov7-tiny.pt --batch-size 32

# 恢复训练
detector-train --resume runs/train/exp/weights/last.pt

# 多 GPU 训练
detector-train --data data/data.yaml --device 0,1,2,3 --batch-size 64

# 超参数进化
detector-train --data data/data.yaml --evolve
```

**相关文件**:
- [`train_cli.py`](detector/cli/train_cli.py:28) - CLI 入口点
- [`train.py`](detector/train.py) - 训练主逻辑

---

### detector-test

**功能**: 测试/评估目标检测模型。支持多种任务模式：正常测试、速度基准测试、研究模式。

**命令**: `detector-test`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | list | `'ckpoints/yolov7-tiny.pt'` | 模型权重文件路径 (model.pt) |
| `--data` | str | `'data/coco.yaml'` | 数据集配置文件路径 (*.data) |
| `--batch-size` | int | `32` | 每个图像批次的大小 |
| `--img-size` | int | `1024` | 推理图像尺寸（像素） |
| `--conf-thres` | float | `0.001` | 目标置信度阈值 |
| `--matrix-conf-thresh` | float | `0.15` | 混淆矩阵置信度阈值 |
| `--iou-thres` | float | `0.45` | NMS 的 IoU 阈值 |
| `--task` | str | `'val'` | 任务类型：train, val, test, speed 或 study |
| `--device` | str | `''` | CUDA 设备，如 `0` 或 `0,1,2,3` 或 `cpu` |
| `--single-cls` | bool | `False` | 作为单类别数据集处理 |
| `--augment` | bool | `False` | 增强推理 |
| `--verbose` | bool | `False` | 按类别报告 mAP |
| `--save-txt` | bool | `False` | 将结果保存到 *.txt 文件 |
| `--save-hybrid` | bool | `False` | 将标签 + 预测混合结果保存到 *.txt |
| `--save-conf` | bool | `False` | 在 --save-txt 标签中保存置信度 |
| `--save-json` | bool | `False` | 保存 cocoapi 兼容的 JSON 结果文件 |
| `--project` | str | `'runs/test'` | 保存结果到 project/name |
| `--name` | str | `'exp'` | 保存结果到 project/name |
| `--exist-ok` | bool | `False` | 已存在的 project/name 不递增 |
| `--no-trace` | bool | `False` | 不 trace 模型 |
| `--v5-metric` | bool | `False` | 在 AP 计算中假设最大召回率为 1.0 |

**调用示例**:

```bash
# 验证集测试
detector-test --data data/data.yaml --weights runs/train/exp/weights/best.pt --task val

# 测试集测试
detector-test --data data/data.yaml --weights runs/train/exp/weights/best.pt --task test

# 多模型对比测试
detector-test --data data/data.yaml --weights model1.pt model2.pt model3.pt

# 速度基准测试
detector-test --data data/data.yaml --weights model.pt --task speed

# 研究模式（在不同图像尺寸下测试）
detector-test --data data/coco.yaml --task study --iou 0.65 --weights yolov7.pt
```

**相关文件**:
- [`test_cli.py`](detector/cli/test_cli.py:18) - CLI 入口点
- [`test.py`](detector/test.py) - 测试主逻辑

---

### detector-detect

**功能**: 在图像、视频或流上运行目标检测推理。支持 webcam 输入、结果可视化保存等功能。

**命令**: `detector-detect`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | `'yolov7.pt'` | 模型权重文件路径 |
| `--source` | str | `'inference/images'` | 检测源（文件/文件夹，0 表示摄像头） |
| `--img-size` | int | `1024` | 推理图像尺寸（像素） |
| `--conf-thres` | float | `0.15` | 目标置信度阈值 |
| `--iou-thres` | float | `0.45` | NMS 的 IoU 阈值 |
| `--device` | str | `''` | CUDA 设备，如 `0` 或 `0,1,2,3` 或 `cpu` |
| `--view-img` | bool | `False` | 显示检测结果 |
| `--save-txt` | bool | `False` | 将结果保存到 *.txt 文件 |
| `--save-conf` | bool | `False` | 在保存结果中保存置信度 |
| `--nosave` | bool | `False` | 不保存图像/视频结果 |
| `--classes` | list | `None` | 按类别过滤：`--class 0` 或 `--class 0 2 3` |
| `--agnostic-nms` | bool | `False` | 类别无关的 NMS |
| `--augment` | bool | `False` | 增强推理 |
| `--update` | bool | `False` | 更新所有模型（修复 SourceChangeWarning） |
| `--project` | str | `'runs/detect'` | 保存结果到 project/name |
| `--name` | str | `'exp'` | 保存结果到 project/name |
| `--exist-ok` | bool | `False` | 已存在的项目/名称不递增 |
| `--no-trace` | bool | `False` | 不 trace 模型 |

**调用示例**:

```bash
# 对单张图像进行推理
detector-detect --weights best.pt --source inference/images/bus.jpg

# 对文件夹中的所有图像进行推理
detector-detect --weights best.pt --source inference/images

# 使用摄像头进行实时检测
detector-detect --weights best.pt --source 0 --view-img

# 使用 RTSP 流进行实时检测
detector-detect --weights best.pt --source rtsp://192.168.1.1/stream

# 保存检测结果到文件
detector-detect --weights best.pt --source inference/images --save-txt --save-conf

# 按类别过滤检测结果
detector-detect --weights best.pt --source inference/images --classes 0 2 3
```

**相关文件**:
- [`detect.py`](detector/cli/detect.py:214) - CLI 入口点

---

### detector-show

**功能**: 显示目标检测结果。用于可视化检测输出，将预测结果绘制到图像上。

**命令**: `detector-show`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--result-home` | str | `''` | 预测结果的路径（包含检测结果的 txt 文件） |
| `--img-home` | str | `''` | 图像的路径（原始图像目录） |
| `--conf` | float | `None` | 置信度阈值，低于此值的检测结果将被过滤 |
| `--save-dir` | str | `None` | 保存绘制结果的目录路径 |
| `--imgname` | str | `'images'` | 图像文件夹名称 |

**调用示例**:

```bash
# 显示检测结果
detector-show --result-home ./results --img-home ./data --conf 0.3 --save-dir ./visualizations

# 指定图像文件夹名称
detector-show --result-home ./pred_results --img-home ./dataset --imgname images --save-dir ./output
```

**相关文件**:
- [`show_detector_results.py`](detector/show_detector_results.py:110) - CLI 入口点

---

### detector-predict

**功能**: 对图像目录进行批量检测，输出 YOLO 格式的检测结果文件。支持单文件夹模式和多文件夹模式。

**命令**: `detector-predict`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | `'weights/best.pt'` | 模型权重文件路径 (.pt) |
| `--home` | str | `None` | 包含多个图像文件夹的根目录（多文件夹模式） |
| `--voc` | str | `None` | 单个图像文件夹路径（单文件夹模式） |
| `--batch-size` | int | `64` | 每批次的图像数量 |
| `--img-size` | int | `1024` | 推理图像尺寸（像素） |
| `--conf-thres` | float | `0.01` | 目标置信度阈值 |
| `--iou-thres` | float | `0.45` | NMS 的 IoU 阈值 |
| `--save-dir` | str | `None` | 检测结果保存目录 |
| `--rgbname` | str | `'images'` | 图像文件夹名称 |
| `--augment` | bool | `False` | 使用增强推理模式（多尺度推理） |

**调用示例**:

```bash
# 单文件夹模式
detector-predict --weights runs/train/exp/weights/best.pt \
    --voc /path/to/images \
    --batch-size 32 \
    --img-size 1024 \
    --conf-thres 0.01

# 多文件夹模式
detector-predict --weights runs/train/exp/weights/best.pt \
    --home /path/to/data_home \
    --batch-size 32 \
    --img-size 1024

# 指定输出目录
detector-predict --weights best.pt --voc ./data/images --save-dir ./results
```

**相关文件**:
- [`predict_bbox.py`](detector/cli/predict_bbox.py:21) - CLI 入口点
- [`predict_bbox.py`](detector/predict_bbox.py) - 批量推理主逻辑

---

### compute-anchors

**功能**: 使用 k-means 聚类计算数据集的最优锚框（anchors）。

**命令**: `compute-anchors`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data` | str | 必需 | 数据集配置 YAML 文件路径 |
| `--num-anchors` | int | `9` | 要计算的锚框数量 |
| `--img-size` | int | `1024` | 锚框计算的图像尺寸 |
| `--thresh` | float | `5.0` | k-means 的 IoU 阈值 |
| `--iters` | int | `5000` | k-means 迭代次数 |
| `--verbose` | bool | `False` | 打印详细输出 |

**调用示例**:

```bash
# 计算 9 个锚框
compute-anchors --data data/data.yaml --num-anchors 9 --img-size 1024

# 计算 12 个锚框，使用更高的迭代次数
compute-anchors --data data/data.yaml --num-anchors 12 --iters 10000 --verbose
```

**相关文件**:
- [`compute_anchors.py`](detector/cli/compute_anchors.py:51) - CLI 入口点
- [`autoanchor.py`](detector/utils/autoanchor.py) - 锚框计算逻辑

---

### print-anchors

**功能**: 从训练好的模型权重文件中读取并打印锚框。

**命令**: `print-anchors`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | 必需 | 模型权重文件路径 (.pt) |

**调用示例**:

```bash
# 打印模型中的锚框
print-anchors --weights runs/train/exp/weights/best.pt
```

**输出示例**:
```
12.50,15.30  25.60,32.40  45.20,58.90
...
```

**相关文件**:
- [`print_anchors.py`](detector/cli/print_anchors.py:45) - CLI 入口点

---

### detector-trim

**功能**: 从训练好的模型中裁剪（移除）指定的类别。用于减少模型大小或创建特定类别的模型。

**命令**: `detector-trim`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | 必需 | 模型权重文件路径 (.pt) |
| `--to-trim` | str | 必需 | 要裁剪的类别索引，逗号分隔（例如：`"3,4"`） |
| `--out-file` | str | `None` | 输出文件路径（默认：权重目录带 trim 后缀） |
| `--force` | bool | `False` | 无需确认直接覆盖输出文件 |

**调用示例**:

```bash
# 裁剪类别 3 和 4
detector-trim --weights runs/train/exp/weights/best.pt --to-trim "3,4"

# 指定输出文件路径
detector-trim --weights best.pt --to-trim "0,1,2" --out-file /path/to/trimmed.pt

# 强制覆盖已存在的输出文件
detector-trim --weights best.pt --to-trim "5" --out-file /path/to/trimmed.pt --force
```

**相关文件**:
- [`model_trim.py`](detector/cli/model_trim.py:148) - CLI 入口点

---

### detector-rename

**功能**: 重命名训练好的模型中的类别名称。

**命令**: `detector-rename`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | 必需 | 模型权重文件路径 (.pt) |
| `--newnames` | str | 必需 | 新的类别名称，逗号分隔（例如：`"car,person,dog"`） |
| `--out-file` | str | `None` | 输出文件路径（默认：权重目录中的 model_newname.pt） |

**调用示例**:

```bash
# 重命名类别
detector-rename --weights runs/train/exp/weights/best.pt --newnames "car,person,dog,cat,bird"

# 指定输出文件
detector-rename --weights best.pt --newnames "class1,class2,class3" --out-file /path/to/renamed.pt
```

**相关文件**:
- [`model_rename_classes.py`](detector/cli/model_rename_classes.py:61) - CLI 入口点

---

### export-onnx

**功能**: 将 PyTorch 模型导出为 ONNX 格式，支持动态轴、NMS 集成、模型简化等功能。

**命令**: `export-onnx`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | 必需 | 模型权重文件路径 |
| `--img-size` | list | `[640, 640]` | 图像尺寸（高，宽） |
| `--batch-size` | int | `1` | 批次大小 |
| `--dynamic` | bool | `False` | 动态 ONNX 轴 |
| `--dynamic-batch` | bool | `False` | 动态批次大小（用于 TensorRT 和 ONNX Runtime） |
| `--grid` | bool | `False` | 导出 Detect() 层网格 |
| `--device` | str | `'cpu'` | CUDA 设备，如 `0` 或 `0,1,2,3` 或 `cpu` |
| `--simplify` | bool | `False` | 简化 ONNX 模型 |
| `--include-nms` | bool | `False` | 导出带 NMS 的端到端 ONNX |
| `--opset` | int | `12` | ONNX opset 版本 |
| `--topk-all` | int | `100` | 每张图像的前 K 个目标 |
| `--iou-thres` | float | `0.45` | NMS 的 IoU 阈值 |
| `--conf-thres` | float | `0.25` | NMS 的置信度阈值 |
| `--max-wh` | int | `None` | TensorRT NMS 为 None，ONNX Runtime NMS 为整数值 |

**调用示例**:

```bash
# 基本导出
export-onnx --weights runs/train/exp/weights/best.pt

# 导出带动态批次大小
export-onnx --weights best.pt --dynamic-batch

# 导出带 NMS 的端到端模型
export-onnx --weights best.pt --include-nms --opset 13

# 简化 ONNX 模型
export-onnx --weights best.pt --simplify --dynamic

# TensorRT 导出
export-onnx --weights best.pt --grid --include-nms --max-wh None --topk-all 100
```

**相关文件**:
- [`export_onnx.py`](detector/cli/export_onnx.py:180) - CLI 入口点

---

### export-fake-rgbt

**功能**: 将 PyTorch 模型导出为 Fake RGBT ONNX 格式，支持 RGB + IR 双输入。

**命令**: `export-fake-rgbt`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | 必需 | 模型权重文件路径 |
| `--img-size` | list | `[640, 640]` | 图像尺寸（高，宽） |
| `--batch-size` | int | `1` | 批次大小 |
| `--dynamic` | bool | `False` | 动态 ONNX 轴 |
| `--dynamic-batch` | bool | `False` | 动态批次大小 |
| `--grid` | bool | `False` | 导出 Detect() 层网格 |
| `--device` | str | `'cpu'` | CUDA 设备 |
| `--simplify` | bool | `False` | 简化 ONNX 模型 |
| `--include-nms` | bool | `False` | 导出带 NMS 的端到端 ONNX |
| `--opset` | int | `12` | ONNX opset 版本 |

**调用示例**:

```bash
# 基本导出
export-fake-rgbt --weights runs/train/exp/weights/best.pt

# 导出带动态批次大小
export-fake-rgbt --weights best.pt --dynamic-batch

# 简化模型
export-fake-rgbt --weights best.pt --simplify --img-size 1024 1024
```

**相关文件**:
- [`export_fake_rgbt.py`](detector/cli/export_fake_rgbt.py:175) - CLI 入口点

---

### export-scale-conf

**功能**: 将 PyTorch 模型导出为带缩放置信度的 ONNX 格式，支持网格模式和非网格模式。

**命令**: `export-scale-conf`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | 必需 | 模型权重文件路径 |
| `--img-size` | list | `[1024, 1024]` | 图像尺寸（高，宽） |
| `--batch-size` | int | `1` | 批次大小 |
| `--dynamic` | bool | `False` | 动态 ONNX 轴 |
| `--dynamic-batch` | bool | `False` | 动态批次大小 |
| `--grid` | bool | `False` | 导出 Detect() 层网格 |
| `--end2end` | bool | `False` | 导出端到端 ONNX |
| `--scale-conf` | float | `1.0` | 置信度缩放因子 |
| `--device` | str | `'cpu'` | CUDA 设备 |
| `--simplify` | bool | `False` | 简化 ONNX 模型 |
| `--include-nms` | bool | `False` | 导出带 NMS 的端到端 ONNX |
| `--opset` | int | `12` | ONNX opset 版本 |
| `--max-wh` | int | `None` | TensorRT NMS 为 None，ONNX Runtime NMS 为整数值 |
| `--topk-all` | int | `100` | 每张图像的前 K 个目标 |
| `--iou-thres` | float | `0.45` | NMS 的 IoU 阈值 |
| `--conf-thres` | float | `0.25` | NMS 的置信度阈值 |

**调用示例**:

```bash
# 基本导出（默认 scale_conf=1.0）
export-scale-conf --weights runs/train/exp/weights/best.pt

# 指定置信度缩放因子
export-scale-conf --weights best.pt --scale-conf 0.5

# 网格模式导出
export-scale-conf --weights best.pt --grid --scale-conf 0.8

# 端到端导出
export-scale-conf --weights best.pt --end2end --include-nms
```

**相关文件**:
- [`export_scale_conf.py`](detector/cli/export_scale_conf.py:256) - CLI 入口点

---

### detector-analyze

**功能**: 分析检测结果中的假阳性（FP）和假阴性（FN），支持可视化绘制和错误列表生成。

**命令**: `detector-analyze`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | list | `'runs/train/5cls_6/weights/best.pt'` | 模型权重文件路径 |
| `--data` | str | 必需 | 数据列表 txt 文件路径 |
| `--batch-size` | int | `32` | 每个图像批次的大小 |
| `--img-size` | int | `1024` | 推理图像尺寸（像素） |
| `--conf-thres` | float | `0.15` | 目标置信度阈值 |
| `--iou-thres` | float | `0.45` | NMS 的 IoU 阈值 |
| `--augment` | bool | `False` | 增强推理 |
| `--no-trace` | bool | `False` | 不 trace 模型 |
| `--save-dir` | str | `None` | 保存结果的目录路径 |
| `--prefix-path` | str | `None` | 数据文件前缀路径 |
| `--draw-path` | str | `'here'` | 保存绘制图像的路径 |
| `--no-draw` | bool | `False` | 不绘制图像 |
| `--no-compare` | bool | `False` | 不在图像中绘制真实标签 |
| `--write-tp` | bool | `False` | 绘制正确预测并保存真阳性图像 |
| `--error-list` | str | `None` | 保存错误列表文件的路径 |

**调用示例**:

```bash
# 基本分析
detector-analyze --weights runs/train/exp/weights/best.pt --data test_list.txt

# 保存检测结果
detector-analyze --weights best.pt --data test_list.txt --save-dir ./results

# 绘制分析结果
detector-analyze --weights best.pt --data test_list.txt --draw-path ./visualizations

# 生成错误列表
detector-analyze --weights best.pt --data test_list.txt --error-list errors.txt

# 完整分析（绘制 + 对比 + 保存 TP）
detector-analyze --weights best.pt --data test_list.txt \
    --draw-path ./viz \
    --write-tp \
    --error-list errors.txt
```

**相关文件**:
- [`analyze_list.py`](detector/cli/analyze_list.py:311) - CLI 入口点

---

### detector-generate-templates

**功能**: 生成模板配置文件（data.yaml, cfg.yaml, hyp.yaml）到指定目录，用于快速创建训练配置。

**命令**: `detector-generate-templates`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output`, `-o` | str | 必需 | 输出目录，用于放置模板文件 |
| `--force` | bool | `False` | 如果文件已存在则覆盖 |

**调用示例**:

```bash
# 生成模板到指定目录
detector-generate-templates --output ./my_project

# 覆盖已存在的文件
detector-generate-templates --output ./my_project --force
```

**生成的文件**:
- `data.yaml` - 数据集配置文件（基于 `data/data_test.yaml`）
- `cfg.yaml` - 模型配置文件（基于 `cfg/yolov7-tiny-silu-sqnet.yaml`）
- `hyp.yaml` - 超参数配置文件（基于 `data/hyp.finetune.tiny.yaml`）

**使用生成的模板**:

```bash
# 使用生成的模板文件进行训练
detector-train --data ./my_project/data.yaml --cfg ./my_project/cfg.yaml --hyp ./my_project/hyp.yaml
```

**相关文件**:
- [`generate_templates.py`](detector/cli/generate_templates.py:22) - CLI 入口点

---

## 命令速查表

| 命令 | 功能 | 主要用途 |
|------|------|----------|
| `detector-train` | 模型训练 | 训练新的目标检测模型 |
| `detector-test` | 模型测试 | 评估模型性能 |
| `detector-detect` | 目标检测 | 对图像/视频进行推理 |
| `detector-show` | 显示结果 | 可视化检测输出 |
| `detector-predict` | 批量推理 | 对图像目录进行批量检测 |
| `compute-anchors` | 计算锚框 | 使用 k-means 计算最优锚框 |
| `print-anchors` | 打印锚框 | 从模型中提取锚框 |
| `detector-trim` | 裁剪类别 | 移除模型中的指定类别 |
| `detector-rename` | 重命名类别 | 修改模型类别名称 |
| `export-onnx` | 导出 ONNX | 转换为 ONNX 格式 |
| `export-fake-rgbt` | 导出 RGBT | 转换为双输入 ONNX |
| `export-scale-conf` | 导出缩放 | 带缩放置信度的 ONNX |
| `detector-analyze` | 分析结果 | 分析 FP/FN 错误 |
| `detector-generate-templates` | 生成模板 | 快速创建训练配置文件 |

---

## 常见问题

### 1. 如何查看命令的帮助信息？

所有命令都支持 `--help` 参数：

```bash
detector-train --help
```

### 2. 如何使用 GPU 进行推理？

使用 `--device` 参数指定 GPU 设备：

```bash
detector-detect --weights best.pt --source image.jpg --device 0
```

### 3. 如何导出 ONNX 模型？

使用以下参数组合：

```bash
export-onnx --weights best.pt --grid
```

### 4. 如何恢复中断的训练？

使用 `--resume` 参数：

```bash
detector-train --resume runs/train/exp/weights/last.pt
```

---

## 版本信息

- **包版本**: 0.1.0
- **Python 要求**: >= 3.7
- **主要依赖**: PyTorch >= 1.7.0, torchvision >= 0.8.0

---

## 相关链接

- [项目源码](https://github.com/example/detector-train)
- [问题追踪](https://github.com/example/detector-train/issues)
