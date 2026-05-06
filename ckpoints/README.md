# Pre-trained Weights Directory

此目录用于存放预训练权重文件。

## 支持的权重文件

- `yolov7-tiny.pt` - YOLOv7-tiny 预训练权重
- `yolov7.pt` - YOLOv7 预训练权重
- `best.pt` - 训练得到的最优模型权重
- `last.pt` - 训练得到的最后一个 epoch 的权重

## 下载预训练权重

从官方 YOLOv7 仓库下载预训练权重：

```bash
# YOLOv7-tiny
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt

# YOLOv7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

## 使用说明

安装 package 时，此目录中的 `.pt` 文件将自动包含在安装包中。

训练和测试命令默认使用 `ckpoints/yolov7-tiny.pt` 作为初始权重。
