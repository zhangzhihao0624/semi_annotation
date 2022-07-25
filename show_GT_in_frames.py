import cv2
import numpy as np
import os
from random import randint




#bounding box yolo format to voc format
def yolo2voc(bboxes, image_height=480, image_width=640):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y2]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes


# give Ground Truth folder and image folder,then draw GT bounding box in each frame
def showGT(GT_folder,image_folder):
    GTlist=os.listdir(GT_folder)
    GTlist.sort(key=lambda x:int(x[6:-4]))
    imglist = os.listdir(image_folder)
    imglist.sort(key=lambda x:int(x[6:-4]))
    # print(GTlist)
    img_num=len(imglist)
    GT_num=len(GTlist)
    # print(img_num,GT_num)
    for j in range(img_num):
        GT_path=os.path.join(GT_folder,GTlist[j])
        with open(GT_path,'r') as f:
            data=f.read()
            data=data.split()
            data=list(map(float,data))
        object_num=int(len(data)/5)
        bbox_list=[]
        for i in range(object_num):
            bbox_list.append([data[1+i*5],data[2+i*5],data[3+i*5],data[4+i*5]])

        bbox_list=np.array(bbox_list)
        img_path=os.path.join(image_folder,imglist[j])
        image=cv2.imread(img_path)
        # print(image.shape)
        for i in range(object_num):
            colors=[randint(0,255),randint(0,255),randint(0,255)]
            a=bbox_list[i]
            bbox=yolo2voc(a,image_height=480,image_width=640)
            x1=int(bbox[0])
            y1=int(bbox[1])
            x2=int(bbox[2])
            y2=int(bbox[3])
            cv2.rectangle(image, (x1,y1),(x2,y2), colors,2)
        cv2.imshow('Ground Truth', image)
        cv2.waitKey(80) & 0xFF
    cv2.destroyAllWindows()

GT_folder='/home/zzh/hand_object_detector/GT_2d'
image_folder='/home/zzh/hand_object_detector/images'
showGT(GT_folder,image_folder)
