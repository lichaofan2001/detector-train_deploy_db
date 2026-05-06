# 主要运行程序
import os
import numpy as np
import logging

def split_path(pn, level=1):
    ''' 将路径进行分割，level是子路径的层级
    例：对于路径 a/b/c/d/e/f，不同level的返回值如下
    level=1, a/b/c/d/e  f
    levle=2  a/b/c/d    e/f
    level=3  a/b/c      d/e/f
    '''
    pa, name = os.path.split(pn)
    for _ in range(level-1):
        if(not pa):
            break
        pa, nameX = os.path.split(pa)

        name = os.path.join(nameX, name)
    return pa, name

def read_det_file(detfile):
    ''' 读取预测结果，返回标签，检测框，置信度
    标签为字符串格式列表
    检测框为浮点数列表，每个检测框为4个浮点数，xyxy格式
    置信度为浮点数列表
    '''
    pd_labels = []
    pd_boxes = []
    pd_scores = []
    with open(detfile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if(line):
                items = line.split(' ')
                pd_labels.append(items[0])
                pd_scores.append(float(items[1]))
                pd_boxes.append([float(items[2]), float(items[3]), float(items[4]), float(items[5])])
    return pd_labels, pd_boxes, pd_scores

def get_relative_path(path, base_path):
    path = os.path.abspath(path)
    base_path = os.path.abspath(base_path)
    
    return os.path.relpath(path, base_path)

# 根据图像文件名，生成检测结果文件名
def imgfile2detfile(imgfile, det_path_name='prediction', uppper_level=3, save_dir=None, prefix_path = None):
    img_pn, filename = os.path.split(imgfile)
    voc, _ = os.path.split(img_pn)

    if(save_dir is None):
        det_pn = os.path.join(voc, det_path_name)
    elif prefix_path is not None:
        num_prefix = len(prefix_path)
        if(voc == prefix_path):
            if(det_path_name is None):
                det_pn = save_dir
            else:
                det_pn = os.path.join(save_dir, det_path_name) 
        else:
            if(prefix_path[-1]=='/'):
                num_prefix -= 1
            if(voc[num_prefix]=='/'):
                num_prefix += 1
            name = voc[num_prefix:]
            if(det_path_name is not None):
                det_pn = os.path.join(save_dir, name, det_path_name)
            else:
                det_pn = os.path.join(save_dir, name)
    else:
        _, name = split_path(voc, level=uppper_level)
        if det_path_name is not None:
            det_pn = os.path.join(save_dir, name, det_path_name)
        else:
            det_pn = os.path.join(save_dir)
    #if(not os.path.exists(det_pn)):
    #    os.makedirs(det_pn)

    ext = filename.split('.')[-1]
    Len_ext = len(ext)
    return os.path.join(det_pn, filename[:-Len_ext]+f'txt'), det_pn

# 根据图像文件名，生成标注文件名
def imgfile2annofile(imgfile, anno_path_name='annotations'):
    img_pn, filename = os.path.split(imgfile)
    voc, _ = os.path.split(img_pn)
    annopn = os.path.join(voc, anno_path_name)

    ext = filename.split('.')[-1]
    Len_ext = len(ext)
    return os.path.join(annopn, filename[:-Len_ext]+f'xml')

def read_list(test_file):
    if(not os.path.exists(test_file)):
        return []
    with open(test_file, 'r') as f:
        test_list = []
        for line in f.readlines():
            line = line.strip()
            if(line):
                test_list.append(line)
        return test_list

class DetPath(object):
    def __init__(self, det_file_folder, pid:int):
        self.det_file_folder = det_file_folder
        self.pid = pid

    def format(self, classname):
        return self.get_full_file(classname)
    
    def get_filename(self, classname):
        return f'{self.pid}-{classname}.txt'
    
    def get_full_file(self, classname):
        return os.path.join(self.det_file_folder, self.get_filename(classname))
    
def check_path_exist(pn):
    if(os.path.exists(pn)):
        return 
    
    pa, _ = os.path.split(pn)
    if(pa):
        check_path_exist(pa)
    os.makedirs(pn)
    
class AnnoPath(object):
    # 根据图像绝对路径，查找得到标注的路径
    def format(self, imgname):
        return imgfile2annofile(imgname)


class ResultIO(object):
    def __init__(self, det_file_folder:str, prd_name:str, keyparams:dict, test_file:str=None, custuom_name=None):
        
        self.keyparams = keyparams
        
        if(det_file_folder is None): # 检测文件不设置时，默认保存在和test_file同级的目录下
            pn, filename = os.path.split(test_file)
            basename = filename.split('.')[0]
            if(custuom_name is None):
                det_file_folder = os.path.join(pn, f'detections_{basename}_{self.get_indentify_str()}')
            else:
                det_file_folder = os.path.join(pn, f'{custuom_name}_detections_{basename}_{self.get_indentify_str()}')
        
        self.det_file_folder = det_file_folder
        self.result_file = os.path.join(det_file_folder, f'detection_metrics_result-{prd_name}.txt')
        
        pid = os.getpid()
        self.det_path = DetPath(det_file_folder, pid)
        
    def get_det_path(self): #-> DetPath:
        return self.det_path

    def get_result_file(self):
        return self.result_file
    
    def get_indentify_str(self):
        keyparam_str = '_'.join([f'{k}_{v}' for k, v in self.keyparams.items()])
        
        return keyparam_str
        
    def write(self, classnames:list, recalls:list, precisions:list, aps:list):

        logging.info(f'Write TO > {self.result_file}')
        if(not os.path.exists(self.det_file_folder)):
            os.makedirs(self.det_file_folder)
        
        keyparam_str = ', '.join([f'{k}={v}' for k, v in self.keyparams.items()])
        with open(self.result_file, 'w') as f:
            f.write(f'Classname, Recall, Precision, AP, with {keyparam_str}\n')
            for i in range(len(classnames)):
                f.write(f'{classnames[i]}, {recalls[i]}, {precisions[i]}, {aps[i]} \n')
                
    def read(self):
        classnames = []
        recalls = []
        precisions = []
        aps = []
        
        if(not os.path.exists(self.result_file)):
            logging.warning(f'File {self.result_file} not exists')
            return classnames, recalls, precisions, aps
        with open(self.result_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                classname, recall, precision, ap = line.strip().split(', ')
                classnames.append(classname)
                recalls.append(float(recall))
                precisions.append(float(precision))
                aps.append(float(ap))
        return classnames, recalls, precisions, aps
