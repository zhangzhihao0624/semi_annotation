import cv2
import numpy as np
import os


def iou(a,b):

    area_a = a[2] * a[3]
    area_b = b[2] * b[3]

    w = min(b[0]+b[2],a[0]+a[2]) - max(a[0],b[0])
    h = min(b[1]+b[3],a[1]+a[3]) - max(a[1],b[1])

    if w <= 0 or h <= 0:
        return 0

    area_c = w * h

    return area_c / (area_a + area_b - area_c)

def yolo2coco(bboxes, image_height, image_width):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y2]
    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    w = bboxes[2] * image_width
    h = bboxes[3] * image_height

    x_min = bboxes[0]*image_width - w / 2
    y_min = bboxes[1]*image_height - h/2

    bboxes=[x_min,y_min,w,h]

    return bboxes

def get_GT_bboxes(GT_path,image_height,image_width):
    # GT_path='/home/zzh/RAFT/GT_2d/frame_0.txt'
    with open(GT_path, 'r') as f:
        data = f.read()
        data = data.split()
        data = list(map(float, data))
    object_num = int(len(data) / 5)
    bbox_list = []
    for i in range(object_num):
        bbox_list.append([data[0+5*i:5+5*i]])

    bbox_list = np.array(bbox_list)
    GT = []
    for i in range(len(bbox_list)):
        a = yolo2coco(bbox_list[i][0, 1:5], image_height, image_width)
        a.insert(0, bbox_list[i][0, 0])
        GT.append(a)

    GT = np.array(GT)
    return GT

# GT_folder='s1_t5_t4_test_GT'
def Caculate_accuracy(GT_folder,Track_result_path,image_path):
    GT_list=os.listdir(GT_folder)
    GT_list.sort(key=lambda x: int(x[6:-4]))
    GT_num=len(GT_list)
    # print(GT_list)
    track=np.load(file=Track_result_path)
    # print(track[0,::])
    # print(track.shape)
    # for k in track[0, ::]:
    #     print(k[0])
    im=cv2.imread(image_path)
    height=im.shape[0]
    width=im.shape[1]
    # GT_path=os.path.join(GT_folder,GT_list[0])
    # GT=get_GT_bboxes(GT_path,image_height=height,image_width=width)
    # print(GT)
    # # print(GT)
    TP = 0
    FP = 0

    iou_list=[]
    for i in range(GT_num-1):
        GT_path=os.path.join(GT_folder,GT_list[i])
        GT=get_GT_bboxes(GT_path,image_height=height,image_width=width)
        track_bboxs=track[i , ::]
        for s in GT:
            GT_box=yolo2coco(s[1:5],image_height=height,image_width=width)
            for j in track_bboxs:
                track_box=yolo2coco(j[1:5],image_height=height,image_width=width)
                if s[0]==j[0]:
                    if iou(GT_box,track_box)>0.5:
                        TP+=1
                        iou_list.append(iou(GT_box,track_box))
                    else:
                        FP+=1
                        iou_list.append(iou(GT_box, track_box))

    acc=TP/(TP+FP)
    data_iou=np.mean(iou_list)
    return TP, FP, acc, data_iou



GT_folder='s1_t5_t4_test_GT'
Track_result_path='s1_t5_t5_pro.npy'
image_path='s1_t5_t4_testfolder/237.png'
print(Caculate_accuracy(GT_folder,Track_result_path,image_path))





