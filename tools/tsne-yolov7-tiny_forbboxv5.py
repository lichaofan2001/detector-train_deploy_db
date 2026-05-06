#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
需要安装包conda install scikit-learn
python tsne-yolov7-tiny_forbboxv3.py     --weights temp1/model_3cls.pt     --data data/data_0912.yaml --img_size 640

增加功能
1. 背景类框（随机背景/模型疑似输出）
2. 支持多个特征层输出

解决图像不在同一路径的问题，@20251216

'''
#!/usr/bin/env python
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import cv2
import argparse
import yaml
import random 
from detector.utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr

def parse_arguments():
    parser = argparse.ArgumentParser(description='Feature Visualization for YOLOv7-tiny BBoxes')
    
    # Model & input
    parser.add_argument('--weights', type=str, required=True, help='Path to .pt model weights')
    parser.add_argument('--data', type=str, default='', help='Path to dataset YAML (has "val" key)')
    parser.add_argument('--val_txt', type=str, default='', help='Path to val.txt (list of image paths)')
    parser.add_argument('--img_size', type=int, default=1024, help='Input image size (default: 640)')
    
    # Output & config
    parser.add_argument('--class_names', type=str, nargs='+', 
                        default=['Himars_launcher', 'Lt2000_launcher', 'Tb_launcher'])
    parser.add_argument('--feature_layers', type=int, nargs='+', default=[58, 66, 74])
    parser.add_argument('--output_image', type=str, default='tsne_result')
    parser.add_argument('--output_npz', type=str, default='bbox_features_results.npz')
    #parser.add_argument('--target_dim', type=int, default=256)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--save_results', action='store_true', default=True)

    return parser.parse_args()

# 图像路径转换为标签路径
def imgpath2labelpath(folder:str, label_folder_name='labels'):
    pa, _ = os.path.split(folder)
    if(folder.endswith('/')):
        pa, _ = os.path.split(pa)
    return os.path.join(pa, label_folder_name)

def imgfile2labelfile(imgfile, label_folder_name='labels'):
    imgpath, imgname = os.path.split(imgfile)
    labelpath = imgpath2labelpath(imgpath)
    #labelname = ''.join(imgname.split('.')[:-1]) + '.txt'
    
    ext = os.path.splitext(imgname)[1]
    labelname = imgname[:-len(ext)] + '.txt'
    return os.path.join(labelpath, labelname)


def get_image_paths_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    val_path = data['val']
    if not os.path.isabs(val_path):
        val_path = os.path.join(os.path.dirname(yaml_path), val_path)
    with open(val_path, 'r') as f:
        rel_paths = [line.strip() for line in f if line.strip()]
    img_dir = os.path.dirname(val_path)
    return [os.path.join(img_dir, p) for p in rel_paths]

def get_pred_bbox(out):
    out = non_max_suppression(out, conf_thres=0.01, iou_thres=0.1, labels=None, multi_label=True, agnostic=True)[0]

    xyxy = out[:, :4]
    
    return xyxy 

def compute_iou(xyxy1, xyxy2):
    
    x11, y11, x12, y12 = xyxy1 
    x21, y21, x22, y22 = xyxy2
    
    x1 = max(x11, x21)
    y1 = max(y11, y21)
    
    x2 = min(x12, x22)
    y2 = min(y12, y22)
    
    if(x2<=x1 or y2<=y1):
        iou = 0
    else:
        comman_area = (y2-y1)*(x2-x1)
        a1 = (y12-y11)*(x12 - x11)
        a2 = (y22-y21)*(x22 - x21)
        
        iou = comman_area /(a1+a2-comman_area)
        
    return iou

def get_random_shift_box(bboxes, nmax=3, dist_min = 10, dist_max=100):
    ''' 对图像随机平移得到的背景框
    '''
    
    random_boxes = []
    
    n = len(bboxes)
    
    if(n==0):
        return random_boxes
    if(n>nmax):
        ixs = random.choices(range(n), k=nmax)
    else:
        ixs = range(n)
        
    for ix in ixs:
        bbox = bboxes[ix]
        x1, y1, x2, y2 = bbox['bbox']
        
        xshift = random.random()*(dist_max-dist_min) + dist_min
        yshift = random.random()*(dist_max-dist_min) + dist_min
        
        w = x2-x1 
        h = y2-y1
        
        x1n, y1n, x2n, y2n = x1 - xshift -w, y1-yshift-h, x2-xshift-w, y2-yshift-h 
        
        if(x2n>w and y2n>h):
            
            random_boxes.append([x1n, y1n, x2n, y2n])
            
    
    return random_boxes
            
            
            
        
        

def get_non_overlaped_bbox(box_xyxy, bboxes, iou_thresh=0.01):
    ''' 得到不重叠的框
    '''
    
    boxes = []
    
    random_bboxes =  get_random_shift_box(bboxes, nmax=10)
    
    for xyxy in list(box_xyxy) + random_bboxes:
        
        have_overlap = False 
        for bbox in bboxes:
            bbox = bbox['bbox']
            
            # 计算IOU           
            iou = compute_iou(xyxy, bbox)
            

            if(iou>iou_thresh):
                have_overlap = True 
                break 
        if(not have_overlap):
            boxes.append(xyxy)
            
    return boxes 


def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(args.weights, map_location=device)
    model = ckpt['model'] if 'model' in ckpt else ckpt
    model = model.float().to(device).eval() # 推理模式
    print(f"Model loaded from {args.weights}")

    # Get image list
    if args.data:
        image_paths = get_image_paths_from_yaml(args.data)
        print(f"Loaded {len(image_paths)} images from YAML: {args.data}")
    elif args.val_txt:
        with open(args.val_txt, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(image_paths)} images from val.txt: {args.val_txt}")
    else:
        raise ValueError("Either --data or --val_txt must be provided")


    # Register hooks
    features_dict = {}
    def hook_fn(name):
        def hook(module, inp, out):
            features_dict[name] = out[0] if isinstance(out, tuple) else out
        return hook
    for i, layer_idx in enumerate(args.feature_layers):
        model.model[layer_idx].register_forward_hook(hook_fn(f'layer_{layer_idx}'))
    print(f"Registered hooks on layers: {args.feature_layers}")

    # Preprocess & label parsing helpers
    def preprocess(img_path, target_size=640):
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        original_h, original_w = img.shape[:2]
        scale = min(target_size / original_w, target_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Padding 
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_top = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = resized_img

        # Convert to RGB and normalize

        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        

        return img_tensor, (original_w, original_h, scale, pad_left, pad_top)

    def parse_label(label_path, original_w, original_h, scale, pad_left, pad_top):
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc_norm, yc_norm, bw_norm, bh_norm = map(float, parts[1:5])
                        
                        # Convert normalized to original pixels
                        xc_orig = xc_norm * original_w
                        yc_orig = yc_norm * original_h
                        bw_orig = bw_norm * original_w
                        bh_orig = bh_norm * original_h


                        xc_scaled = xc_orig * scale
                        yc_scaled = yc_orig * scale
                        bw_scaled = bw_orig * scale
                        bh_scaled = bh_orig * scale

           
                        xc_canvas = xc_scaled + pad_left
                        yc_canvas = yc_scaled + pad_top

                        x1 = xc_canvas - bw_scaled / 2
                        y1 = yc_canvas - bh_scaled / 2
                        x2 = xc_canvas + bw_scaled / 2
                        y2 = yc_canvas + bh_scaled / 2

                        bboxes.append({'class_id': cls_id, 'bbox': [x1, y1, x2, y2]})
        return bboxes

    def extract_roi_feat(fmap, bbox):
        x1, y1, x2, y2 = bbox
        _, C, H, W = fmap.shape
        sx, sy = W / args.img_size, H / args.img_size
        fx1, fy1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        fx2, fy2 = min(W, int(x2 * sx)), min(H, int(y2 * sy))
        if fx2 <= fx1 or fy2 <= fy1:
            return None
        roi = fmap[:, :, fy1:fy2, fx1:fx2]
        if roi.numel() == 0:
            return None
        pooled = torch.nn.functional.adaptive_avg_pool2d(roi, (1,1))
        return pooled.view(C).cpu().numpy()

    # Main loop
    all_feats, all_labels, all_info = {}, {}, {}
    success_count = 0

    for i, img_path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"Processing {i}/{len(image_paths)}")

        tensor, orig_info = preprocess(img_path, args.img_size)
        if tensor is None: continue
        original_w, original_h, scale, pad_left, pad_top = orig_info
        label_path = imgfile2labelfile(img_path)
        bboxes = parse_label(label_path, original_w, original_h, scale, pad_left, pad_top)
        if not bboxes: continue

        with torch.no_grad():
            features_dict.clear()
            out, _ = model(tensor)
            
            box_xyxy = get_pred_bbox(out)
            box_xyxy = get_non_overlaped_bbox(box_xyxy, bboxes)
            
            
            for bbox_info in bboxes:
                cls_id = bbox_info['class_id']
                bbox = bbox_info['bbox']
                feat = None
                for name, fmap in features_dict.items():
                    feat = extract_roi_feat(fmap, bbox)
                    if feat is not None:
                        if(name not in all_info):
                            all_info[name] = []
                            all_feats[name] = []
                            all_labels[name] = []
                            
                        all_info[name].append({'image': img_path, 'bbox': bbox, 'class_id': cls_id, 'layer': name})
                        all_feats[name].append(feat)
                        all_labels[name].append(cls_id)
                        
            # 处理随机的背景box
            
            cls_id = len(args.class_names)
            for bbox in box_xyxy:
                
                for name, fmap in features_dict.items():
                    feat = extract_roi_feat(fmap, bbox)
                    if feat is not None:
                        if(name not in all_info):
                            all_info[name] = []
                            all_feats[name] = []
                            all_labels[name] = []
                            
                        all_info[name].append({'image': img_path, 'bbox': bbox, 'class_id': cls_id, 'layer': name})
                        all_feats[name].append(feat)
                        all_labels[name].append(cls_id)

        success_count += 1

    print(f"Processed {success_count} images")

    if len(all_feats) == 0:
        print("No features extracted. Exiting.")
        return

    for name in all_feats.keys():
        
        
        feats = np.array(all_feats[name])
        labels = np.array(all_labels[name])

        # t-SNE
        if len(feats) > args.max_samples:
            idxs = np.random.choice(len(feats), args.max_samples, replace=False)
            feats, labels = feats[idxs], labels[idxs]

        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, max_iter=1000)
        embed = tsne.fit_transform(feats)

        class_names_with_bk = list(args.class_names) + ['Background']
        # Plot
        plt.figure(figsize=(14, 10))
        colors = [plt.cm.tab10(l) for l in labels]
        plt.scatter(embed[:,0], embed[:,1], c=colors, s=15, alpha=0.7)

        handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i), 
                            markersize=10, label=class_names_with_bk[i]) 
                for i in range(len(class_names_with_bk))]
        plt.legend(handles=handles, title="Classes")
        plt.title(f'Layer {name}: BBox t-SNE ({len(feats)} samples, {len(args.class_names)} classes)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.savefig(f'{args.output_image}.{name}.png', dpi=300, bbox_inches='tight')
        print(f"保存绘图结果> {args.output_image}.{name}.png")

    if args.save_results:
        np.savez(args.output_npz, features=all_feats, labels=all_labels, bbox_info=all_info)
        print(f"保存数据> {args.output_npz}")


if __name__ == '__main__':
    main()
