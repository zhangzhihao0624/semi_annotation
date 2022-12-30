from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import collections
import tkinter
import glob
import mediapipe as mp
import math
import os
import cv2
import numpy as np
import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

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


def DrawROI(first_rgb_frame):
    ini_bboxes = []
    end_bboxes = []
    colors = []

    while True:
        cv2.putText(first_rgb_frame, 'Select target ROI and press ENTER', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        cv2.putText(first_rgb_frame, 'press q to quit DrawROI', (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        bbox = cv2.selectROI('Initial box', first_rgb_frame, fromCenter=False)
        ini_bboxes.append(bbox)
        # colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

    # while True:
    #     cv2.putText(end_rgb_frame, 'Select target ROI and press ENTER', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
    #                 1, (255, 0, 0), 1)
    #     cv2.putText(end_rgb_frame, 'press q to quit DrawROI', (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
    #                 1, (255, 0, 0), 1)
    #     bbox = cv2.selectROI('End box', end_rgb_frame, fromCenter=False)
    #     end_bboxes.append(bbox)
    #     key = cv2.waitKey(0)
    #     if key == ord("q"):
    #         break
    # cv2.destroyAllWindows()

    # if len(ini_bboxes) != len(end_bboxes):
    #     raise Exception('The number of drawn boxes in initial and end frame must be identical')

        # draw_box_file = "s1_t7_t1_DrawBox.npz"
        # np.savez(record_path + draw_box_file, ini_bboxes = ini_bboxes, end_bboxes = end_bboxes, colors = colors)
    # else:
    #     raise Exception('The number of drawn boxes in initial and end frame must be identical')

    box_num = len(ini_bboxes)
    bbox_final_keys = InputLabel(box_num)
    print('Corresponding labels: {}'.format(bbox_final_keys))
    return ini_bboxes,  bbox_final_keys

def siammask_tracker(ini_box,img_num,imglist):
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    tracker_boxes = np.zeros([4,img_num])

    for frame_idx in range(img_num):
        frame = cv2.imread(os.path.join(args.imageFolder, imglist[frame_idx]))
        # frame = cv2.imread(os.path.join(args.imageFolder, 'frame_' + str(frame_idx) + '.png'))
        if first_frame:
            init_rect = ini_box
            tracker.init(frame, init_rect)
            x1 = ini_box[0]
            y1 = ini_box[1]
            x2 = ini_box[0] + ini_box[2]
            y2 = ini_box[1] + ini_box[3]
            tracker_boxes[:,0] = np.array([x1,y1,x2,y2])
            first_frame = False
        else:
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            x1=bbox[0]
            y1=bbox[1]
            x2=bbox[0]+bbox[2]
            y2=bbox[1]+bbox[3]
            tracker_boxes[:,frame_idx]=np.array([x1,y1,x2,y2])

            # if 'polygon' in outputs:
            #     polygon = np.array(outputs['polygon']).astype(np.int32)
            #     cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
            #                   True, (0, 255, 0), 3)
            #     mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
            #     mask = mask.astype(np.uint8)
            #     mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
            #     frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            # else:
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 255, 0), 3)
            cv2.imshow('SiamMask', frame)
            cv2.waitKey(5)

    cv2.destroyAllWindows()

    return tracker_boxes

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str,default='/home/zhihao/pysot-master/experiments/siammask_r50_l3/config.yaml', help='config file')
    parser.add_argument('--snapshot', type=str,default='/home/zhihao/pysot-master/experiments/siammask_r50_l3/model.pth')
    parser.add_argument('--imageFolder', default='/home/zhihao/pysot-master/s6_t2_t6/rgb', type=str,help='image files')
    parser.add_argument('--GTdir', default='/home/zhihao/pysot-master/s6_t2_t6/obj_train_data', type=str,help='Groundtruth files')
    # parser.add_argument('--flowFolder', default='/home/zhihao/pysot-master/breakfast2/breakfast2_flow', type=str,
    #                     help='Raft flow files')
    # parser.add_argument('--edgebboxes', default='/home/zhihao/pysot-master/breakfast2/breakfast2_edgebboxes.npz', type=str,
    #                     help='Raft flow files')
    # parser.add_argument('--handsbboxes', default='/home/zhihao/pysot-master/breakfast2/breakfast2_handsbboxes.npz', type=str,
    #                     help='Raft flow files')

    args = parser.parse_args()

    imglist = os.listdir(args.imageFolder)
    imglist.sort(key=lambda x: int(x[6:-4]))  # according to the image name choose sorted number

    img_num = len(imglist)

    first_rgb_frame = cv2.imread(os.path.join(args.imageFolder, imglist[0]))
    image_size = first_rgb_frame.shape[:2][::-1]

    ini_bboxes,  bbox_final_keys = DrawROI(first_rgb_frame)
    box_num = len(ini_bboxes)

    final_bboxes = np.zeros([box_num, 4, img_num])

    for box_idx in range(box_num):
        final_bboxes[box_idx,:,:]=siammask_tracker(ini_bboxes[box_idx],img_num,imglist)

    bbox_final_dict = {}
    for i, key in enumerate(bbox_final_keys):
        bbox_final_dict[key] = final_bboxes[i, :, :]

    GT_dict = load_groundtruth(args.GTdir, image_size, box_num, bbox_final_keys)

    if len(bbox_final_dict) == len(GT_dict):
        accuracy_dict, average_iou,TP, FP, FN = CalculateIOU(bbox_final_dict, GT_dict)
        print('TP: {}, FP: {}, FN:{}'.format(TP,FP,FN))
        print('accuracy: {}'.format(accuracy_dict))
        print('average iou: {}'.format(average_iou))
        print('Precision: {}, Recall: {}'.format(TP/(TP+FP),TP/(TP+FN)))
    else:
        raise Exception('number of ground truth labels is not identical to number of drawn bounding boxes')



