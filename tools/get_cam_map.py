import torch 
import cv2 
from detector.utils.datasets import letterbox
import numpy as np
import matplotlib.pyplot as plt
import os 
# 加载模型，计算热力图，估计模型关注的区域

def preprocess_img(img_file, img_size):
    img = cv2.imread(img_file)
    img, ratio, (dw, dh) = letterbox(img, img_size, stride=32)
    
    return img

def img2tensor(img, device):
    img = img[:, :, ::-1].transpose(2, 0, 1) 
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()/255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    return img 

def make_input(img_file, device, img_size):
    
    img = preprocess_img(img_file=img_file, img_size=img_size)
    
    img_tensor = img2tensor(img, device)
    
    return img_tensor, img


class GradCaM:
    def __init__(self, model, layers=[75, 76, 77]):
        self.model = model 
        self.gradients = {}
        self.activations = {}
        
        self._register_hooks(layers)
        
    def _register_hooks(self, layers):
        
        def forward_hook_fn(name):
        
            def forward_hook_0(module, input, output):
                self.activations[name] = output.detach()
                
            return forward_hook_0
        
        def backward_hook_fn(name):
        
            def backward_hook_0(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            
            return backward_hook_0
        
        # [-2, -3, -4]
        for layer in layers:
            module = self.model.model[layer]
            
            module.register_forward_hook(forward_hook_fn(f'layer_{layer}'))
            module.register_full_backward_hook(backward_hook_fn(f'layer_{layer}'))

def get_classes(x):
    
    classes = torch.argmax(x, dim=1)
    N = len(classes)
    return classes, x[range(N), classes]

def visualize_gradcam(img0, cam, alpha=0.5):
    H, W, C = img0.shape 
    
    cam_resized = cv2.resize(cam, (W, H))
    
    heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
    #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap*alpha + img0 * (1-alpha)
    
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def compute_cam_map(model, img_file, args, device, out_dir):
    
    basename = os.path.basename(img_file).rsplit('.')[0]
    

    conf_thresh = args.conf
    class_id = args.class_id
    
    img, img0 = make_input(img_file=img_file, device=device, img_size=args.imgsz)
    
    
    

    
    model.zero_grad()
    
    grad_cam = GradCaM(model, layers = args.layers)
    
    
    pred = model(img)
    
    # 找出与类别相关的预测结果
    loss = 0 

    
    for p in pred:
        p = p.sigmoid()
        
        p = p[0].reshape(-1, 10)
        
        xywhs, obj_scores, classes_scores = torch.split(p, (4, 1, 5), dim=1)
        
        classes, cls_scores = get_classes(classes_scores)
        
        
        obj_scores = obj_scores.flatten()        
        conf_scores = obj_scores * cls_scores
        
        if(class_id is None):
            scores = conf_scores[(conf_scores>conf_thresh)] # any class
        else:    
            scores = conf_scores[(classes==class_id) & (conf_scores>conf_thresh)]
        
        print(scores.detach())
        
        loss += torch.sum(scores)
        
    #print(f'loss= {loss.item()}')
    loss.backward()
    # 计算梯度图
    
    img0_shape = img0.shape[:2]
    
    
    NUM = len(args.layers)
    fig, axs = plt.subplots(NUM, 1)
    
    for ix, k in enumerate(grad_cam.activations.keys()):
        grad = grad_cam.gradients[k][0]
        acti = grad_cam.activations[k][0]
        
        weights = torch.mean(grad, dim=[1, 2])
        
        cam = torch.mean( acti * weights.reshape((-1, 1, 1)), dim=0)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) # 变换为0-1
        cam = cam.detach().cpu().numpy()
        
        superimposed_img0, heat_map = visualize_gradcam(img0, cam, args.ratio)
        cv2.imwrite(os.path.join(out_dir, f'superimposed_img_{basename}_{k}.jpg'), superimposed_img0)
    
        ax = axs[ix]
        ax.imshow(heat_map)
        
        
    layer_ixs = '_'.join([str(x) for x in args.layers])
    fig.savefig( os.path.join(out_dir, f'heatmap_{basename}_{layer_ixs}.png'))
    fig.close()
    

def is_img(fn):
    exts = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    
    for ext in exts:
        if(fn.endswith(ext)):
            return True 
    
    return False

def read_list(fn):
    filelist = []
    with open(fn, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if(line):
                filelist.append(line)
    return filelist


def ensure_path_exist(pn):
    if(os.path.exists(pn)):
        return 
    pa, _ = os.path.split(pn)
    
    if(pa):
        ensure_path_exist(pa)
        
    os.makedirs(pn)



if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='图像列表，jpg文件，txt文件或者包含图像的文件夹', type=str, help='图像路径')
    parser.add_argument('--weights', default='runs/train/malan5cls_2/weights/best.pt', type=str, help='YOLOv7模型权重路径')
    parser.add_argument('--conf', default=0.25, type=float, help='阈值')
    parser.add_argument('--imgsz', default=1024, type=int, help='输入大小')
    parser.add_argument('--class_id', default=None, type=int, help='类型ID, 不限类别')
    
    parser.add_argument('--out-dir', default='YOLOv7-GRAD_CAM', type=str, help='图像输出路径')
    
    parser.add_argument('--layers', default=[75, 76, 77], type=int, nargs='+', help='层数')
    parser.add_argument('--ratio', default=0.15, type=float, help='可视化时GradCam权重比重')
    args = parser.parse_args()
    
    model_file = args.weights
    
    device = torch.device('cuda:0')
    
    model = torch.load(model_file, map_location=device)['model']
    
    model = model.float()
    model = model.to(device)
    model.train()
    
    for k, v in model.named_parameters():
        v.requires_grad = True  # 计算梯度信息
        
        
    source = args.source 
    
    listfile = []
    if(os.path.isdir(source)):
        listfile = [os.path.join(source, fn) for fn in os.listdir(source) if is_img(fn)]
    elif(os.path.isfile(source)):
        if(is_img(source)):
            listfile = [source]
        elif(source.endswith('.txt')):
            listfile = read_list(source)
        else:
            print(f'位置的输入类型 {source}')
            exit(-1)
    else:
        print(f'输入数据错误')
        
    print(f'共需处理{len(listfile)} 个图像')
    
    ensure_path_exist(args.out_dir)

    for img_file in listfile:
        if(os.path.exists(img_file) and is_img(img_file)):
            compute_cam_map(model, img_file, args, device, args.out_dir)