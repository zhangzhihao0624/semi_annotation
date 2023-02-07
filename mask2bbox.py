import tkinter

import cv2
from torchvision.io import read_image,ImageReadMode
from torchvision.utils import draw_bounding_boxes
import numpy as np
from torchvision.ops import masks_to_boxes
import os

def InputLabel(box_num):
    label_list = []
    box_num = box_num

    def GetLabel():
        nonlocal label_list, box_num

        try:
            label_list = var_input.get().strip()
            label_list = list(label_list.split(','))
            if len(label_list) == box_num:
                # for label in label_list:
                #     # str(label.split('_')[0])
                #     # int(label.split('_')[1])
                print('Label input successful')
                root.destroy()
            else:
                var_input.set('The number of labels and drawn boxes must be identical')
        except:
            var_input.set('Input failed, labels should be defined like above')

    root = tkinter.Tk(className='Specify labels')
    root.geometry('600x120')

    show_text = tkinter.Label(root, text="Specify labels of bounding boxes like: cup, bowl, milk")
    show_text.pack(pady=20)

    var_input = tkinter.StringVar()
    label_entry = tkinter.Entry(root, textvariable=var_input, width=45)
    label_entry.pack(side='left', expand=True)

    input_button = tkinter.Button(root, text='Input', command=GetLabel)
    input_button.pack(side='left', expand=True, ipadx=20)

    root.mainloop()
    return label_list

def yolo2voc(bboxes, image_height, image_width):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y2]
    """
    # bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    w = bboxes[2] * image_width
    h = bboxes[3] * image_height

    x_min=bboxes[0]*image_width-w/2
    y_min=bboxes[1]*image_height-h/2

    bboxes=[x_min,y_min,x_min+w,y_min+h]

    return np.array(bboxes)


def load_groundtruth(GT_dir,image_size,box_num,bbox_final_keys):
    width = image_size[0]
    height = image_size[1]

    GT_list = os.listdir(GT_dir)
    GT_list.sort(key=lambda x: int(x[6:-4]))
    image_num = len(GT_list)

    GT_bboxes=np.zeros([box_num,4,image_num])

    for image_idx in range(image_num):
        GT_path = os.path.join(GT_dir, GT_list[image_idx])
        with open(GT_path, 'r') as f:
            data = f.read()
            data = data.split()
            data = list(map(float, data))
        for box_idx in range(box_num):
            box=[data[1+box_idx*5],data[2+box_idx*5],data[3+box_idx*5],data[4+box_idx*5]]
            GT_bboxes[box_idx, :, image_idx]= yolo2voc(box,height,width)


    GT_dict={}
    for i, key in enumerate(bbox_final_keys):
        GT_dict[key] = GT_bboxes[i, :, :]

    return GT_dict

def CalculateIOU(bbox_final_dict, GT_dict):
    accuracy_dict = {}
    TP = 0
    FP = 0
    FN = 0
    for bbox_final_key in bbox_final_dict:
        img_num = GT_dict[bbox_final_key].shape[1]
        iou_list = []
        for img_idx in range(img_num):
            # calculate accuracy only if there is ground truth
            if np.all(GT_dict[bbox_final_key][:,img_idx])!=0:
                bbox_dx = bbox_final_dict[bbox_final_key][2,img_idx] - bbox_final_dict[bbox_final_key][0,img_idx]
                bbox_dy = bbox_final_dict[bbox_final_key][3,img_idx] - bbox_final_dict[bbox_final_key][1,img_idx]
                bbox_area = (bbox_dx + 1)*(bbox_dy + 1)
                GT_dx = GT_dict[bbox_final_key][2,img_idx] - GT_dict[bbox_final_key][0,img_idx]
                GT_dy = GT_dict[bbox_final_key][3,img_idx] - GT_dict[bbox_final_key][1,img_idx]
                GT_area = (GT_dx + 1)*(GT_dy + 1)

                x_left = max(bbox_final_dict[bbox_final_key][0,img_idx], GT_dict[bbox_final_key][0,img_idx])
                y_top = max(bbox_final_dict[bbox_final_key][1,img_idx], GT_dict[bbox_final_key][1,img_idx])
                x_right = min(bbox_final_dict[bbox_final_key][2,img_idx], GT_dict[bbox_final_key][2,img_idx])
                y_bottom = min(bbox_final_dict[bbox_final_key][3,img_idx], GT_dict[bbox_final_key][3,img_idx])

                intersection_area = max(0,(x_right - x_left + 1)) * max(0,(y_bottom - y_top + 1))
                iou = intersection_area / (bbox_area + GT_area - intersection_area)
                iou_list.append(iou)
                if iou>0.5:
                    TP = TP + 1
                else:
                    FP = FP + 1
                if GT_area != 0 and bbox_area == 0:
                    FN = FN + 1
        accuracy_dict[bbox_final_key] = np.mean(iou_list)

    average_iou=np.mean(list(accuracy_dict.values()))

    return accuracy_dict, average_iou,TP, FP, FN


def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
    stats = stats[stats[:,4].argsort()]
    return stats[:-1]


# im=cv2.imread('/home/zhihao/XMem/workspace/coffee_t2/images/0000000.jpg')
# mask = cv2.imread('/home/zhihao/XMem/workspace/cereal_t1/masks/0000000.png',0)
# Lower = np.array([37])
# Upper = np.array([51])
# Binary = cv2.inRange(mask, Lower, Upper)
# cv2.imshow("strawberry", Binary)
# cv2.waitKey(100)
# # cv2.imshow('mask2bbox',mask)
# # cv2.waitKey(0)
# # print(mask)
# # ret, mask = cv2.threshold(mask, 112, 255, cv2.THRESH_TOZERO)
# bboxs = mask_find_bboxs(Binary)
# print(bboxs)

# for b in bboxs:
#     x0, y0 = b[0], b[1]
#     x1 = b[0] + b[2]
#     y1 = b[1] + b[3]
#     cv2.rectangle(im, (x0, y0), (x1, y1), [0, 0, 255], 2, cv2.LINE_AA)
#
# cv2.imshow('mask2bbox',im)
# cv2.waitKey(0)



mask_class={0:[37,51],1:[74,88],2:[112,127],3:[13,37],4:[51,74],5:[88,112],6:[127,255]}
mask_colors={0:[0,0,255],1:[0,255,0],2:[0,255,255],3:[255,0,0],4:[139,72,61],5:[209,206,0],6:[224,255,255]}

image_folder='/home/zhihao/XMem/workspace/s6_t5_t1/images'
mask_folder='/home/zhihao/XMem/workspace/s6_t5_t1/masks'
GTdir='/home/zhihao/XMem/workspace/s6_t5_t1/obj_train_data'
obj_num=6

imglist=os.listdir(image_folder)
imglist.sort(key=lambda x: int(x[0:-4]))

first_rgb_frame = cv2.imread(os.path.join(image_folder, imglist[0]))
image_size = first_rgb_frame.shape[:2][::-1]

masklist=os.listdir(mask_folder)
masklist.sort(key=lambda x: int(x[0:-4]))

# print(imglist)
# print(masklist)


bbox_final_keys=InputLabel(obj_num)
img_num=len(imglist)
# print(img_num)

final_bboxes = np.zeros([obj_num, 4, img_num])

last_framebboxes=[[] for i in range(obj_num)]
for img_idx in range(img_num):
    im=cv2.imread(os.path.join(image_folder,imglist[img_idx]))
    mask=cv2.imread(os.path.join(mask_folder,masklist[img_idx]),0)
    for obj_idx in range(obj_num):
        Lower = np.array([mask_class[obj_idx][0]])
        Upper = np.array([mask_class[obj_idx][1]])
        Binary = cv2.inRange(mask, Lower, Upper)
        bboxs = mask_find_bboxs(Binary)
        if bboxs.any():
            last_framebboxes[obj_idx].append(bboxs)
            box=[]
            for b in bboxs:
                x0, y0 = b[0], b[1]
                x1 = b[0] + b[2]
                y1 = b[1] + b[3]
                box.append((x0,y0))
                box.append((x1,y1))

            box.sort(key=lambda elem: (elem[0], elem[1]))

            cv2.rectangle(im, box[0], box[-1], mask_colors[obj_idx], 2, cv2.LINE_AA)
            final_bboxes[obj_idx,:,img_idx]=np.array([x0,y0,x1,y1])
        else:
            b=last_framebboxes[obj_idx][-1][0]
            x0, y0 = b[0], b[1]
            x1 = b[0] + b[2]
            y1 = b[1] + b[3]
            cv2.rectangle(im, (x0, y0), (x1, y1), mask_colors[obj_idx], 2, cv2.LINE_AA)
            final_bboxes[obj_idx, :, img_idx] = np.array([x0, y0, x1, y1])

    cv2.imshow('mask2bbox',im)
    cv2.waitKey(30)
cv2.destroyAllWindows()

bbox_final_dict = {}
for i, key in enumerate(bbox_final_keys):
    bbox_final_dict[key] = final_bboxes[i, :, :]


GT_dict = load_groundtruth(GTdir, image_size, obj_num, bbox_final_keys)

if len(bbox_final_dict) == len(GT_dict):
    accuracy_dict, average_iou,TP, FP, FN = CalculateIOU(bbox_final_dict, GT_dict)
    print('TP: {}, FP: {}, FN:{}'.format(TP,FP,FN))
    print('Precision: {}, Recall: {}'.format(TP / (TP + FP), TP / (TP + FN)))
    print('accuracy: {}'.format(accuracy_dict))
    print('average iou: {}'.format(average_iou))

else:
    raise Exception('number of ground truth labels is not identical to number of drawn bounding boxes')































# img=read_image('/home/zhihao/XMem/workspace/coffee_t2/images/0000000.jpg',mode=ImageReadMode.RGB)
# mask=read_image('/home/zhihao/XMem/workspace/coffee_t2/masks/0000000.png', mode=ImageReadMode.UNCHANGED)
#
#
# bboxs = masks_to_boxes(mask)
# bboxs=bboxs.numpy()
# print(bboxs,bboxs.shape)
# for box in bboxs:
#     x1,y1,x2,y2=map(int,box)
#     print(x1,y1,x2,y2)
# print(img)
# print(type(img))
# print(img.shape)

# img_with_boxes=draw_bounding_boxes(img,bboxs,colors='red',fill=False,width=1)
# cv2.imshow('mask2bbox',img)
# cv2.waitKey(0)
# img_cv2=img.numpy()
# img_cv2=img_cv2.transpose(1,2,0)
# # print(img_cv2)
# # print(img_cv2.shape)
# img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
# # cv2.imshow('mask2bbox',img_cv2)
# # cv2.waitKey(0)
# im=cv2.imread('/home/zhihao/XMem/workspace/coffee_t2/masks/0000000.png')
# print(im.shape)
# for box in bboxs:
#     x1,y1,x2,y2=map(int,box)
#     # print(x1,y1,x2,y2)
#     cv2.rectangle(im, (x1, y1), (x2, y2), [0,0,255], 2, cv2.LINE_AA)
#
# cv2.imshow('mask2bbox',im)
# cv2.waitKey(0)