import torch 
import cv2 
from detector.utils.datasets import letterbox
import numpy as np
import matplotlib.pyplot as plt
# 加载模型，计算热力图，估计模型关注的区域，对图像进行遮挡

def preprocess_img(img_file, img_size):
    img = cv2.imread(img_file)
    img, ratio, (dw, dh) = letterbox(img, img_size, stride=32)
    
    return img, (img.shape[0], img.shape[1])

def img2tensor(img, device):
    img = img[:, :, ::-1].transpose(2, 0, 1) 
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()/255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    return img 

def make_input(img_file, device, img_size=1024):
    
    img, (H0, W0) = preprocess_img(img_file=img_file, img_size=img_size)
    
    img_tensor = img2tensor(img, device)
    
    return img_tensor, img, (H0, W0)



def get_classes(x):
    
    classes = torch.argmax(x, dim=1)
    N = len(classes)
    return classes, x[range(N), classes]

def to_real_heatmap(heatmap, alpha=0.5):
    
    # 归一化到0-1之间
    heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))

    
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap

def iou(xyxy1, xyxy2):
    x1, y1, x2, y2 = xyxy1 
    x21, y21, x22, y22 = xyxy2 
    
    xl = max(x1, x21)
    xr = min(x2, x22)
    yt = max(y1, y21)
    yb = min(y2, y22)
    
    if(xr>xl and yb>yt):
        area = (yb-yt)*(xr-xl)
    else:
        area = 0
    
    a1 = (y2-y1)*(x2-x1)
    a2 = (y22-y21)*(x22-x21)
    
    return area / (a2+a1-area)
    
    


def compute_score(model,img, cls_id, bbox, orig_conf, iou_thresh=0.5, img_size=1024):
    # 计算某个位置的检测框得分
    H0, W0 = img.shape[:2]
    img, _, _ = letterbox(img, img_size, stride=32) 
    img = img2tensor(img, device)
    
    with torch.no_grad():
        
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.01, 0.45, classes=[cls_id, ], agnostic=False)
        
    det = pred[0]
    if(len(det)==0):
        return orig_conf # 显著性
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (H0, W0)).round()
    candidate = []
    for ix, (*xyxy, conf, _) in enumerate(det):
        #print(ix, conf, xyxy)
        
        x1, y1, x2, y2 = xyxy
        conf = conf.item()
        
        x1 = int(x1.item())
        x2 = int(x2.item())
        y1 = int(y1.item())
        y2 = int(y2.item())
        
        iou_score = iou(bbox, (x1, y1, x2, y2))
        
        
        if(iou_score>iou_thresh):
            candidate.append(conf)
    
    if(len(candidate)==0):
        return orig_conf
    
    return orig_conf - np.max(candidate)

if __name__ == '__main__':
    
    from detector.utils.general import non_max_suppression, scale_coords
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpg', default='dlc_0823_M300_027044.jpg', type=str, help='图像路径')
    parser.add_argument('--weights', default='runs/train/malan5cls_2/weights/best.pt', type=str, help='YOLOv7模型权重路径')
    parser.add_argument('--conf', default=0.25, type=float, help='阈值')
    parser.add_argument('--img_size', default=1024, type=int, help='图像大小')
    parser.add_argument('--block_size', default=64, type=int, help='掩码大小')

    parser.add_argument('--class_id', nargs='+',default=None, type=int, help='类型ID, 如果为None则不限类别')
    
    
    parser.add_argument('--index', default=None, type=int, help='测试第几个目标框，如果不指定，根据推理结果，交互式输入')
    
    
    
    
    args = parser.parse_args()
    
    img_file = args.jpg
    model_file = args.weights 
    
    conf_thresh = args.conf
    class_id = args.class_id
    
    device = torch.device('cuda:0')
    
    model = torch.load(model_file, map_location=device)['model']
    
    model = model.float()
    img, img0, (H0, W0) = make_input(img_file=img_file, device=device)
    
    model = model.to(device)
    
    model.eval()
    
    for k, v in model.named_parameters():
        v.requires_grad = False  # 计算梯度信息
    
    
    with torch.no_grad():
        
        pred = model(img)[0]
        
        
        pred = non_max_suppression(pred, args.conf, 0.45, classes=args.class_id, agnostic=False)
    
    # 找出与类别相关的预测结果
    det = pred[0]
    if(len(det)==0):
        print(f'未检测出目标，程序退出！')
        exit(0)

    # 绘制显示图像
    
    IMG = cv2.imread(img_file)
    
    H0, W0 = IMG.shape[:2]
    

        
        
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (H0, W0)).round()
    
    
    
    for ix, (*xyxy, conf, cls_id) in enumerate(det):
        #print(ix, conf, xyxy)
        
        cls_id = int(cls_id.cpu().item())
        txt = f'{ix}: {cls_id}@{conf.cpu().item():.2f}'
        
        x1, y1, x2, y2 = xyxy
        
        x1 = int(x1.item())
        x2 = int(x2.item())
        y1 = int(y1.item())
        y2 = int(y2.item())
        
        
        cv2.rectangle(IMG, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(IMG, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        
        print(txt)
            
            
    cv2.imwrite('out.jpg', IMG)
    
    k = args.index 
    
    while(k is None or k<0 or k>=len(det)):
        k = input('请输入要检测的ID号>\n')
        try:
            k = int(k)
        except:
            k = None 
    
    # 获得对应检测框
    *xyxy, conf, cls_id = det[k]
    
    (x1, y1, x2, y2) = xyxy
    
    x1 = int(x1.item())
    x2 = int(x2.item())
    y1 = int(y1.item())
    y2 = int(y2.item())
    
    bbox = det[k]
    orig_conf = conf.item() 
    cls_id = int(cls_id.item())
    
    # 只计算检测框周围的置信度变化
    
    n_xticks = len(range(x1-64, x2+64, args.block_size))
    n_yticks = len(range(y1-64, y2+64, args.block_size))
    
    heat_map = np.zeros((H0, W0)).astype(np.float32)
    
    for i, x in enumerate(range(x1-64, x2+64, args.block_size)):
        for j, y in enumerate(range(y1-64, y2+64, args.block_size)):
            
            
            x = max(0, x)
            y = max(0, y)
            x = min(W0, x)
            y = min(H0, y)
            
            x_right = min(W0, x + args.block_size)
            y_right = min(H0, y + args.block_size)
            
            if(y_right>y and x_right>x):
            
                IMG = cv2.imread(img_file)
                IMG[y:y_right, x:x_right, :] = 127
                score_diff = compute_score(model, IMG, cls_id, (x1, y1, x2, y2), orig_conf, 0.5, img_size=args.img_size)
                
                
                print(score_diff)
                heat_map[y:y_right, x:x_right] = score_diff
                
            
    alpha = 0.15
    # 将两个热力图进行融合
    heat_map = to_real_heatmap(heat_map)
    
    cv2.imwrite('occ_heatmap.jpg', heat_map)
    
    IMG = cv2.imread(img_file)
    superimposed_img = IMG * (1-alpha) + heat_map * alpha 
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    cv2.imwrite('occolution_map.jpg', superimposed_img)