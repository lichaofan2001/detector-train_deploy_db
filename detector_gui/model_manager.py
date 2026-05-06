"""
统一的模型推理接口

为 YOLO 和大模型提供一致的推理接口，简化调用和管理
"""

import os
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import torch

def postprocess_and_draw(bboxes: List[List[float]], original_shape: Tuple[int, int],
                         classnames: Optional[List[str]] = None,
                         ignore_classes: Optional[List[int]] = None,
                         line_thickness: int = 2,
                         original_image: Any = None) -> Tuple[List[List[float]], Any, int]:
    """
    统一的后处理函数：过滤边界框、绘制标注、生成标签数据

    Args:
        bboxes: 边界框列表，格式为 [[cx, cy, w, h, conf, cls], ...] 或 [[cx, cy, w, h], ...]
        original_shape: 原始图像尺寸 (H, W)
        classnames: 类别名称列表
        ignore_classes: 需要忽略的类别ID列表
        line_thickness: 绘制线条粗细
        original_image: 原始图像数据（用于绘制）

    Returns:
        (filtered_bboxes, plotted_image, det_count)
        - filtered_bboxes: 过滤后的边界框列表（YOLO格式归一化）
        - plotted_image: 绘制了边界框的图像
        - det_count: 检测到的目标数量
    """
    if ignore_classes is None:
        ignore_classes = []

    imgH, imgW = original_shape[:2]

    if original_image is not None:
        plotted = original_image.copy()
    else:
        plotted = np.zeros((imgH, imgW, 3), dtype=np.uint8)

    filtered_bboxes = []
    det_count = 0

    print(f'[调试 postprocess] 收到 {len(bboxes)} 个边界框，ignore_classes: {ignore_classes}')

    for bbox in bboxes:
        if len(bbox) < 4:
            continue

        cx, cy, w, h = bbox[:4]
        conf = bbox[4] if len(bbox) > 4 else 1.0
        cls = int(bbox[5]) if len(bbox) > 5 else 0

        print(f'[调试 postprocess] 处理 bbox: cx={cx:.4f}, cy={cy:.4f}, w={w:.4f}, h={h:.4f}, conf={conf:.4f}, cls={cls}')

        # 检查是否需要忽略该类别
        if cls in ignore_classes:
            print(f'[调试 postprocess] 跳过类别 {cls}')
            continue

        # 转换为像素坐标
        x1 = int((cx - w / 2) * imgW)
        y1 = int((cy - h / 2) * imgH)
        x2 = int((cx + w / 2) * imgW)
        y2 = int((cy + h / 2) * imgH)

        print(f'[调试 postprocess] 像素坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}')

        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(imgW - 1, x2)
        y2 = min(imgH - 1, y2)

        # 获取类别名称
        label = classnames[cls] if classnames and cls < len(classnames) else str(cls)

        # 绘制边界框
        cv2.rectangle(plotted, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
        cv2.putText(plotted, f'{label}:{conf:.2f}', (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f'[调试 postprocess] 绘制完成，label={label}')

        # 添加到过滤后的列表
        filtered_bboxes.append([cx, cy, w, h, conf, cls])
        det_count += 1

    print(f'[调试 postprocess] 最终 det_count={det_count}')
    return filtered_bboxes, plotted, det_count


def save_detection_results(bboxes: List[List[float]], txt_path: str,
                          classnames: Optional[List[str]] = None,
                          ignore_classes: Optional[List[int]] = None) -> int:
    """
    保存检测结果到 YOLO 格式的标签文件

    Args:
        bboxes: 边界框列表
        txt_path: 输出标签文件路径
        classnames: 类别名称列表
        ignore_classes: 需要忽略的类别ID列表

    Returns:
        检测到的目标数量
    """
    if ignore_classes is None:
        ignore_classes = []

    det_count = 0

    with open(txt_path, 'w') as f:
        for bbox in bboxes:
            if len(bbox) < 4:
                continue

            cx, cy, w, h = bbox[:4]
            conf = bbox[4] if len(bbox) > 4 else 1.0
            cls = int(bbox[5]) if len(bbox) > 5 else 0

            # 检查是否需要忽略该类别
            if cls in ignore_classes:
                continue

            # 获取类别名称
            label = classnames[cls] if classnames and cls < len(classnames) else str(cls)

            # 写入 YOLO 格式
            f.write(f'{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n')
            det_count += 1

    return det_count


def convert_xyxy_to_yolo(xyxy: Tuple[float, float, float, float],
                         img_shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """
    将 xyxy 格式的边界框转换为 YOLO 归一化格式

    Args:
        xyxy: (x1, y1, x2, y2) 格式的边界框
        img_shape: 图像尺寸 (H, W)

    Returns:
        (cx, cy, w, h) YOLO 归一化格式
    """
    x1, y1, x2, y2 = xyxy
    imgH, imgW = img_shape[:2]

    cx = (x1 + x2) / 2.0 / imgW
    cy = (y1 + y2) / 2.0 / imgH
    w = (x2 - x1) / imgW
    h = (y2 - y1) / imgH

    return cx, cy, w, h


class BaseModelManager(ABC):
    """模型管理器抽象基类"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.classnames = None

    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass

    @abstractmethod
    def inference(self, image: Any, **kwargs) -> List[List[float]]:
        """
        执行推理

        Args:
            image: 图像路径或图像数据
            **kwargs: 其他推理参数

        Returns:
            List of bounding boxes in format [[cx, cy, w, h, conf, cls], ...]
            or [[cx, cy, w, h], ...] if conf/cls not available
        """
        pass

    @abstractmethod
    def unload_model(self):
        """卸载模型，释放内存"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'classnames': self.classnames,
        }


class YOLOModelManager(BaseModelManager):
    """YOLO 模型管理器"""

    def __init__(self, model_path: str, img_size: int = 640, conf_thres: float = 0.25,
                 iou_thres: float = 0.45, device: str = 'cuda'):
        super().__init__(model_path)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._device_str = device
        self._device = None
        self.stride = 32

    def load_model(self):
        """加载 YOLO 模型"""
        import torch
        from detector.models.experimental import attempt_load
        from detector.utils.general import check_img_size
        from detector.utils.torch_utils import select_device

        self._device = select_device(self._device_str)
        self.model = attempt_load(self.model_path, map_location=self._device)
        self.stride = max(int(self.model.stride.max()), 32)

        # 尝试从 opt.yaml 读取 img_size
        self._try_load_img_size_from_opt_yaml()

        self.img_size = check_img_size(self.img_size, s=self.stride)
        self.model.eval()

        if self._device.type != 'cpu':
            self.model.half()
        else:
            self.model.float()

        self.classnames = self._load_classnames()

        return self

    def _try_load_img_size_from_opt_yaml(self):
        """尝试从 opt.yaml 文件读取 img_size 参数"""
        import yaml
        # 权重路径: C:\...\test\weights\best.pt
        # opt.yaml 路径: C:\...\test\opt.yaml (weights 的上层目录)
        weights_dir = os.path.dirname(self.model_path)  # C:\...\test\weights
        parent_dir = os.path.dirname(weights_dir)        # C:\...\test
        opt_yaml_path = os.path.join(parent_dir, 'opt.yaml')

        print(f'[调试] 尝试从 opt.yaml 读取 img_size: {opt_yaml_path}')

        if os.path.exists(opt_yaml_path):
            try:
                with open(opt_yaml_path, 'r', encoding='utf-8') as f:
                    opt_data = yaml.safe_load(f)

                if opt_data and 'img_size' in opt_data:
                    img_size_val = opt_data['img_size']
                    print(f'[调试] opt.yaml 中的 img_size: {img_size_val}')

                    if isinstance(img_size_val, list):
                        self.img_size = img_size_val[0]
                    elif isinstance(img_size_val, int):
                        self.img_size = img_size_val
                    else:
                        print(f'[调试] img_size 类型不支持: {type(img_size_val)}，使用默认值 1024')
                        self.img_size = 1024

                    print(f'[调试] 从 opt.yaml 设置 img_size 为: {self.img_size}')
                else:
                    print(f'[调试] opt.yaml 中没有 img_size 键，使用默认值 1024')
                    self.img_size = 1024
            except Exception as e:
                print(f'[调试] 读取 opt.yaml 失败: {e}，使用默认值 1024')
                self.img_size = 1024
        else:
            print(f'[调试] opt.yaml 不存在: {opt_yaml_path}，使用默认值 1024')
            self.img_size = 1024

    def _load_classnames(self) -> Optional[List[str]]:
        """加载类别名称"""
        classnames_file = os.path.join(os.path.dirname(self.model_path), 'classnames.txt')
        if os.path.exists(classnames_file):
            with open(classnames_file, encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        return None

    def inference(self, image: Any, **kwargs) -> List[List[float]]:
        """
        YOLO 模型推理

        Args:
            image: 图像路径或图像数据
            **kwargs: 可覆盖默认参数的置信度阈值、IOU阈值等

        Returns:
            [[cx, cy, w, h, conf, cls], ...]
        """
        conf_thres = kwargs.get('conf_thres', self.conf_thres)
        iou_thres = kwargs.get('iou_thres', self.iou_thres)

        if isinstance(image, str):
            img0 = cv2.imread(image)
        else:
            img0 = image.copy()

        original_shape = img0.shape[:2]

        # 图像预处理
        from detector.utils.datasets import letterbox
        img = letterbox(img0, new_shape=self.img_size, stride=self.stride, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._device.type != 'cpu' else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 模型推理
        with torch.no_grad():
            pred = self.model(img)[0]

        # NMS 后处理
        from detector.utils.general import non_max_suppression, scale_coords
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        bboxes = []
        if pred[0] is not None and len(pred[0]):
            det = pred[0].clone()
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_shape).round()

            for *xyxy, conf, cls in det.tolist():
                cx, cy, w, h = convert_xyxy_to_yolo(xyxy, original_shape)
                bboxes.append([cx, cy, w, h, conf, int(cls)])

        print(f'[调试 inference] 返回 {len(bboxes)} 个边界框')
        return bboxes

    def unload_model(self):
        """卸载 YOLO 模型"""
        if self.model is not None:
            del self.model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model = None

    def get_model_info(self) -> Dict[str, Any]:
        """获取 YOLO 模型信息"""
        return {
            'model_path': self.model_path,
            'classnames': self.classnames,
            'img_size': self.img_size,
            'stride': self.stride,
            'conf_thres': self.conf_thres,
            'iou_thres': self.iou_thres,
        }


class LargeVisionModelManager(BaseModelManager):
    """大视觉模型管理器 (如 Qwen)"""

    def __init__(self, model_path: str, prompt: str = None,
                 min_pixels: int = 64 * 32 * 32, max_pixels: int = 9800 * 32 * 32):
        super().__init__(model_path)
        self.prompt = prompt or 'Locate every instance that belongs to the following categories: "military vehicle", Report bbox coordinates in JSON format.'
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.classnames = ['military vehicle']

    def load_model(self):
        """加载大视觉模型"""
        from detector_gui.qwen_manager import Qwen35Manager
        self.model = Qwen35Manager(self.model_path)
        return self

    def inference(self, image: Any, **kwargs) -> List[List[float]]:
        """
        大视觉模型推理

        Args:
            image: 图像路径
            **kwargs: 可覆盖默认参数

        Returns:
            [[cx, cy, w, h], ...]  大模型没有置信度和类别
        """
        if isinstance(image, str):
            image_path = image
        else:
            raise ValueError("大模型只支持图像路径，不支持直接传入图像数据")

        bboxes = self.model.inference(image_path, self.prompt, self.min_pixels, self.max_pixels)

        # 转换为统一格式 [[cx, cy, w, h, conf, cls], ...]
        # 大模型没有置信度和类别，默认为 1.0 和 0
        result = []
        for bbox in bboxes:
            if len(bbox) >= 4:
                result.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), 1.0, 0])

        return result

    def unload_model(self):
        """卸载大视觉模型"""
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model = None

    def get_model_info(self) -> Dict[str, Any]:
        """获取大视觉模型信息"""
        return {
            'model_path': self.model_path,
            'classnames': self.classnames,
            'prompt': self.prompt,
            'min_pixels': self.min_pixels,
            'max_pixels': self.max_pixels,
        }


# 工厂函数：统一创建模型管理器
def create_model_manager(model_type: str, model_path: str, **kwargs) -> BaseModelManager:
    """
    工厂函数：创建模型管理器

    Args:
        model_type: 'yolo' 或 'large'
        model_path: 模型路径
        **kwargs: 其他参数

    Returns:
        BaseModelManager 实例
    """
    if model_type == 'yolo':
        return YOLOModelManager(
            model_path,
            img_size=kwargs.get('img_size', 1024),
            conf_thres=kwargs.get('conf_thres', 0.25),
            iou_thres=kwargs.get('iou_thres', 0.45),
            device=kwargs.get('device', 'cuda' if __import__('torch').cuda.is_available() else 'cpu')
        )
    elif model_type == 'large':
        return LargeVisionModelManager(
            model_path,
            prompt=kwargs.get('prompt'),
            min_pixels=kwargs.get('min_pixels', 64 * 32 * 32),
            max_pixels=kwargs.get('max_pixels', 9800 * 32 * 32)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 统一推理函数
def run_inference_unified(model_manager: BaseModelManager, image: Any,
                         ignore_classes: Optional[List[int]] = None,
                         **kwargs) -> Tuple[List[List[float]], Any, int]:
    """
    统一推理函数：执行推理、后处理、绘制

    Args:
        model_manager: 模型管理器实例
        image: 图像路径或数据
        ignore_classes: 需要忽略的类别ID列表
        **kwargs: 推理参数

    Returns:
        (bboxes, plotted_image, det_count)
    """
    # 执行推理
    bboxes = model_manager.inference(image, **kwargs)

    # 读取原始图像用于可视化
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()

    original_shape = img.shape[:2]

    # 统一后处理
    filtered_bboxes, plotted, det_count = postprocess_and_draw(
        bboxes, original_shape,
        classnames=model_manager.classnames,
        ignore_classes=ignore_classes,
        original_image=img
    )

    return filtered_bboxes, plotted, det_count
