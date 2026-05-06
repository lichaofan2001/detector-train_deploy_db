import os
import glob
import numpy as np
import time
import argparse
import re 
import cv2

import shutil

# 指定预测文件夹与图像文件夹，绘制结果

def show_bbox_prd(im_fp, lb_fp, dstdir, conf_thresh=None):
    
    if(not os.path.exists(dstdir)):
        os.mkdir(dstdir)
    img = cv2.imread(im_fp)
    _, basename = os.path.split(im_fp)
    
    im_h, im_w, _ = img.shape
    
    if(not os.path.exists(lb_fp)):
        cv2.imwrite(os.path.join(dstdir, basename), img)
        return 

    try:
        with open(lb_fp, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if(line):
                    clss, xc, yc, w, h, conf = line.split(' ')
                    roi_cx_f = float(xc)
                    roi_cy_f = float(yc)
                    roi_w_f = float(w)
                    roi_h_f = float(h)
                    conf = float(conf)
                    
                    if(conf_thresh is not None):
                        if(conf<conf_thresh):
                            continue

                    bbox_p0, bbox_p1 = get_bbox(im_w, im_h, roi_cx_f, roi_cy_f, roi_w_f, roi_h_f)
                    cv2.rectangle(img, (bbox_p0[0], bbox_p0[1]), (bbox_p1[0], bbox_p1[1]), (255, 0, 0), 2)
                    
                    txt_code = f'{clss} @ {conf:.2f}'
                    cv2.putText(img, txt_code, (bbox_p0[0], bbox_p0[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2)


            cv2.imwrite(os.path.join(dstdir, basename), img)
    except Exception as e:
        print(f'{lb_fp} 文件读取错误，请查改 {e} ')


def get_bbox(im_w, im_h, roi_cx_f, roi_cy_f, roi_w_f, roi_h_f):
    x0 = int((roi_cx_f - (roi_w_f / 2)) * im_w)
    x1 = int(x0 + roi_w_f * im_w)
    y0 = int((roi_cy_f - (roi_h_f / 2)) * im_h)
    y1 = int(y0 + roi_h_f * im_h)

    p0 = (x0, y0)
    p1 = (x1, y1)

    return p0, p1

def is_img(filename:str):
    exts = ['jpg', 'png', 'jpeg', 'bmp']
    for ext in exts:
        if(filename.lower().endswith(ext)):
            return True 
    return False 

def get_img_list(jpg_dir):
    
    result = []
    for filename in os.listdir(jpg_dir):
        if(is_img(filename)):
            result.append(os.path.join(jpg_dir, filename))
    return result

def write_effect(result_dir, jpg_dir, save_dir, conf_thresh=None):
    
    
    jpg_list = get_img_list(jpg_dir)


    for jpg_path in jpg_list:
        
        basename = os.path.basename(jpg_path)
        ext = os.path.splitext(basename)[1]
        label_name = basename.replace(ext, '.txt')
        

        #print(basename, ext, label_name)
    
        txt_path = os.path.join(result_dir, label_name)
        
        show_bbox_prd(jpg_path, txt_path, save_dir, conf_thresh)


def check_dir_exist(dir):
    if(os.path.exists(dir)):
        return 
    pn, _ = os.path.split(dir)
    if(pn):
        check_dir_exist(pn) # 递归检查父目录是否存在
    os.makedirs(dir)


def show_command():
    """Command line entry point for detector-show command."""
    parser = argparse.ArgumentParser(description='Show detector results')
    parser.add_argument('--result-home', default='', type=str, help='预测结果的路径')
    parser.add_argument('--img-home', default='', type=str, help='图像的路径')
    parser.add_argument('--conf', default=None, type=float, help='阈值')
    parser.add_argument('--save-dir', default=None, type=str, help='预测文件所保存的路径')
    parser.add_argument('--imgname', default='images', type=str, help='预测文件所保存的路径')
    args = parser.parse_args()

    imgname = args.imgname
    home = args.img_home
    result_home = args.result_home
    # 遍历 img-home 下所有以 imgname 为名字的文件夹
    for pn, dirs, _ in os.walk(home):
        for dir in dirs:
            if(dir == imgname):
                imgdir = os.path.join(pn, dir)

                rel_path = os.path.relpath(pn, home)
                result = os.path.join(result_home, rel_path)
                save_dir = os.path.join(args.save_dir, rel_path)

                check_dir_exist(save_dir)   

                write_effect(result, imgdir, save_dir, args.conf)


if __name__ == "__main__":
    show_command()
