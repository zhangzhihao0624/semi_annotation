from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import collections
import tkinter

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

def CalculateBoxCenter(box):
    return np.array([int(box[0] + box[2]/2), int(box[1] + box[3]/2)])

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


def DrawROI(first_rgb_frame, end_rgb_frame):
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

    while True:
        cv2.putText(end_rgb_frame, 'Select target ROI and press ENTER', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        cv2.putText(end_rgb_frame, 'press q to quit DrawROI', (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        bbox = cv2.selectROI('End box', end_rgb_frame, fromCenter=False)
        end_bboxes.append(bbox)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

    if len(ini_bboxes) != len(end_bboxes):
        raise Exception('The number of drawn boxes in initial and end frame must be identical')

        # draw_box_file = "s1_t7_t1_DrawBox.npz"
        # np.savez(record_path + draw_box_file, ini_bboxes = ini_bboxes, end_bboxes = end_bboxes, colors = colors)
    # else:
    #     raise Exception('The number of drawn boxes in initial and end frame must be identical')

    box_num = len(ini_bboxes)
    bbox_final_keys = InputLabel(box_num)
    print('Corresponding labels: {}'.format(bbox_final_keys))
    return ini_bboxes, end_bboxes,  bbox_final_keys

def CalculateBoxCenter(box):
    return np.array([int(box[0] + box[2]/2), int(box[1] + box[3]/2)])

def iou(a,b):

    area_a = a[2] * a[3]
    area_b = b[2] * b[3]

    w = min(b[0]+b[2],a[0]+a[2]) - max(a[0],b[0])
    h = min(b[1]+b[3],a[1]+a[3]) - max(a[1],b[1])

    if w <= 0 or h <= 0:
        return 0

    area_c = w * h

    return area_c / (area_a + area_b - area_c)

def get_RP_boxes(im,num_boxes=200,p_alpha=0.7,p_beta=0.9,p_eta=1.0,p_min_area=150,gray=False):
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('/home/zhihao/RAFT/model.yml')
    #rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if gray == True:
        im_rgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    else:
        im_rgb = im
        edges = edge_detection.detectEdges(np.float32(im_rgb) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(num_boxes)
    edge_boxes.setAlpha(p_alpha)
    edge_boxes.setBeta(p_beta)
    edge_boxes.setEta(p_eta)
    edge_boxes.setMinBoxArea(p_min_area)
    #edge_boxes.setMinScore(0.05)
    #edge_boxes.setGamma(1)
    '''
    # Previous parameters:
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(num_boxes)
    edge_boxes.setAlpha(0.7)
    edge_boxes.setBeta(0.9)
    edge_boxes.setEta(1.0)
    edge_boxes.setMinBoxArea(150)
    '''

    result = edge_boxes.getBoundingBoxes(edges, orimap)
    # for b in boxes[0]:
    #     x, y, w, h = b
    #     cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    boxes,score=result[0],result[1]
    return boxes,score

def startframes_tuple(frames, box_idx):
    if len(frames[box_idx])>2:
        tuple_first = frames[box_idx][0]
        tuple_first_idx = 0
        list_tuple = []
        for i, frame_id in enumerate(frames[box_idx]):
            if i>0:
                tuple_first = frames[box_idx][i-1]
            if frame_id - tuple_first>2:
                list_tuple.append(tuple(frames[box_idx][tuple_first_idx:i]))
                tuple_first_idx = i
            if i == len(frames[box_idx])-1:
                list_tuple.append(tuple(frames[box_idx][tuple_first_idx:]))
        # print(list_tuple)
        # longest_tuple = max(len(tuple_element) for tuple_element in list_tuple)
        # for tuple_element in list_tuple:
        #     if len(tuple_element) == longest_tuple:
        #         max_img_idx = tuple_element[-1]
        # frames[box_idx] = list_tuple[0][0]
        return list_tuple
    elif len(frames[box_idx])==2:
        frames[box_idx] = frames[box_idx][0]
    elif len(frames[box_idx])==1:
        frames[box_idx] = frames[box_idx][0]
    return frames[box_idx]

def endframes_tuple(frames, box_idx):
    if len(frames[box_idx])>2:
        tuple_first = frames[box_idx][0]
        tuple_first_idx = 0
        list_tuple = []
        for i, frame_id in enumerate(frames[box_idx]):
            if i>0:
                tuple_first = frames[box_idx][i-1]
            if frame_id - tuple_first<-2:
                list_tuple.append(tuple(frames[box_idx][tuple_first_idx:i]))
                tuple_first_idx = i
            if i == len(frames[box_idx])-1:
                list_tuple.append(tuple(frames[box_idx][tuple_first_idx:]))

        return list_tuple
    elif len(frames[box_idx])==2:
        frames[box_idx] = frames[box_idx][0]
    elif len(frames[box_idx])==1:
        frames[box_idx] = frames[box_idx][0]
    return frames[box_idx]

def hand_move_trajectory(handsbboxes,frames_interval):
    interval_start=frames_interval[0]
    interval_end=frames_interval[1]
    handscenter=np.zeros([abs(interval_end-interval_start)+1,2,2])
    if interval_start<interval_end:
        n=0
        for i in range(interval_start,interval_end+1):
            for hand_idx in range(2):
                hand_center=CalculateBoxCenter(handsbboxes[i,hand_idx,:])
                handscenter[n,hand_idx,:]=hand_center
            n+=1
    else:
        n = 0
        for i in range(interval_start, interval_end - 1,-1):
            for hand_idx in range(2):
                hand_center = CalculateBoxCenter(handsbboxes[i, hand_idx, :])
                handscenter[n, hand_idx, :] = hand_center
            n += 1

    return handscenter

def obj_move_trajectory(reference_box,frames_interval):
    interval_start = frames_interval[0]
    interval_end = frames_interval[1]
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    obj_center=np.zeros([abs(interval_end-interval_start)+1,2])
    n=0
    if interval_start<interval_end:
        for frame_idx in range(interval_start,interval_end+1):
            frame=cv2.imread(os.path.join(args.imageFolder,str(frame_idx)+'.jpg'))
            # frame = cv2.imread(os.path.join(args.imageFolder, 'frame_' + str(frame_idx) + '.png'))
            if first_frame:
                init_rect = reference_box
                obj_center[n, :] = CalculateBoxCenter(init_rect)
                tracker.init(frame, init_rect)
                first_frame = False
            else:
                outputs = tracker.track(frame)
                bbox = list(map(int, outputs['bbox']))
                obj_center[n,:]=CalculateBoxCenter(bbox)

                if 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                else:
                    bbox = list(map(int, outputs['bbox']))
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 3)
                cv2.imshow('obj center', frame)
                cv2.waitKey(100)
            n+=1
        cv2.destroyAllWindows()
    else:
        for frame_idx in range(interval_start, interval_end -1,-1):
            frame=cv2.imread(os.path.join(args.imageFolder,str(frame_idx)+'.jpg'))
            # frame = cv2.imread(os.path.join(args.imageFolder, 'frame_'+str(frame_idx) + '.png'))
            if first_frame:
                init_rect = reference_box
                obj_center[n, :] = CalculateBoxCenter(init_rect)
                tracker.init(frame, init_rect)
                first_frame = False
            else:
                outputs = tracker.track(frame)
                bbox = list(map(int, outputs['bbox']))
                obj_center[n, :] = CalculateBoxCenter(bbox)

                if 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                else:
                    bbox = list(map(int, outputs['bbox']))
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 3)
                cv2.imshow('obj center', frame)
                cv2.waitKey(100)
            n += 1
        cv2.destroyAllWindows()

    return obj_center


def start_contactframe(ini_bboxes,boxes_num,imglist,edgebboxes,handsbboxes):
    image_num=len(imglist)
    start_frames = [[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        for image_idx in range(image_num-1):
            edge_boxes=edgebboxes[image_idx,:,:]
            hands_boxes=handsbboxes[image_idx,:,:]
            candidate_boxes = []
            # iou filter
            for box in edge_boxes:
                if iou(box, ini_bboxes[box_idx]) > 0.5:
                    candidate_boxes.append([box, iou(box, ini_bboxes[box_idx])])
            candidate_boxes.sort(key=lambda x: x[1], reverse=True)

            if candidate_boxes:
                # hand_preBox_dist=np.zeros([2,1])
                pre_box = candidate_boxes[0][0]
                pre_box_center = CalculateBoxCenter(pre_box)
                for hand_idx in range(2):
                    if iou(hands_boxes[hand_idx],pre_box)>0:
                        start_frames[box_idx].append(image_idx)

    start_tuple=[[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        start_tuple[box_idx]=startframes_tuple(start_frames,box_idx)

    ini_contact=[[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        for tuple_idx in range(len(start_tuple[box_idx])):
            frame_interval=[start_tuple[box_idx][tuple_idx][0],start_tuple[box_idx][tuple_idx][0]+10]
            referencebox=ini_bboxes[box_idx]

            handscenter=hand_move_trajectory(handsbboxes,frame_interval)
            obj_center=obj_move_trajectory(referencebox,frame_interval)

            obj_movingvector = np.diff(obj_center, axis=0)
            average_objmovingvector = np.mean(obj_movingvector, axis=0)
            average_handmovingvector = np.zeros([2, 2])
            for hand_idx in range(2):
                hand_moingvector = np.diff(handscenter[:, hand_idx, :], axis=0)
                average_handmovingvector[hand_idx, :] = np.mean(hand_moingvector, axis=0)

            for hand_idx in range(2):
                v1 = average_objmovingvector
                v2 = average_handmovingvector[hand_idx]
                if np.linalg.norm(v1)<1:
                    continue
                if not(np.any(v2)):
                    continue
                else:
                    cos_sim = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if cos_sim>0.7:
                        ini_contact[box_idx].append(frame_interval[0])
                        break

            if len(ini_contact[box_idx])==1:
                break


    return ini_contact

def end_contactframe(end_bboxes,boxes_num,imglist,edgebboxes,handsbboxes):
    image_num=len(imglist)
    end_frames = [[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        for image_idx in range(image_num-2,-1,-1):
            edge_boxes=edgebboxes[image_idx,:,:]
            hands_boxes=handsbboxes[image_idx,:,:]
            candidate_boxes = []
            # iou filter
            for box in edge_boxes:
                if iou(box, end_bboxes[box_idx]) > 0.5:
                    candidate_boxes.append([box, iou(box, end_bboxes[box_idx])])
            candidate_boxes.sort(key=lambda x: x[1], reverse=True)

            if candidate_boxes:
                # hand_preBox_dist=np.zeros([2,1])
                pre_box = candidate_boxes[0][0]
                pre_box_center = CalculateBoxCenter(pre_box)
                for hand_idx in range(2):
                    if iou(hands_boxes[hand_idx],pre_box)>0:
                        end_frames[box_idx].append(image_idx)
    end_tuple=[[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        end_tuple[box_idx]=endframes_tuple(end_frames,box_idx)

    end_contact = [[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        for tuple_idx in range(len(end_tuple[box_idx])):
            frame_interval = [end_tuple[box_idx][tuple_idx][0], end_tuple[box_idx][tuple_idx][0] - 10]
            referencebox = end_bboxes[box_idx]

            handscenter = hand_move_trajectory(handsbboxes, frame_interval)
            obj_center = obj_move_trajectory(referencebox, frame_interval)

            obj_movingvector = np.diff(obj_center, axis=0)
            average_objmovingvector = np.mean(obj_movingvector, axis=0)
            average_handmovingvector = np.zeros([2, 2])
            for hand_idx in range(2):
                hand_moingvector = np.diff(handscenter[:, hand_idx, :], axis=0)
                average_handmovingvector[hand_idx, :] = np.mean(hand_moingvector, axis=0)

            for hand_idx in range(2):
                v1 = average_objmovingvector
                v2 = average_handmovingvector[hand_idx]
                if np.linalg.norm(v1) < 1:
                    continue
                if not (np.any(v2)):
                    continue
                else:
                    cos_sim = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if cos_sim > 0.7:
                        end_contact[box_idx].append(frame_interval[0])
                        break

            if len(end_contact[box_idx]) == 1:
                break
    return end_contact



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optical flow and Siamtracker demo')
    parser.add_argument('--config', type=str,
                        default='/home/zhihao/pysot-master/experiments/siamrpn_r50_l234_dwxcorr/config.yaml',
                        help='config file')
    parser.add_argument('--snapshot', type=str,
                        default='/home/zhihao/pysot-master/experiments/siamrpn_r50_l234_dwxcorr/model.pth')
    parser.add_argument('--imageFolder', default='/home/zhihao/pysot-master/flower/rgb', type=str,
                        help='image files')
    parser.add_argument('--flowFolder', default='/home/zhihao/pysot-master/flower/flower_flow', type=str,
                        help='Raft flow files')
    parser.add_argument('--edgebboxes', default='/home/zhihao/pysot-master/flower/flower_edgebboxes.npz', type=str,
                        help='Raft flow files')
    parser.add_argument('--handsbboxes', default='/home/zhihao/pysot-master/flower/flower_handsbboxes.npz', type=str,
                        help='Raft flow files')

    args = parser.parse_args()

    imglist = os.listdir(args.imageFolder)
    imglist.sort(key=lambda x: int(x[0:-4]))  # according to the image name choose sorted number

    img_num = len(imglist)

    first_rgb_frame = cv2.imread(os.path.join(args.imageFolder, imglist[0]))
    end_rgb_frame = cv2.imread(os.path.join(args.imageFolder, imglist[-1]))
    # cv2.imshow('test',first_rgb_frame)
    # cv2.waitKey(0)

    flowlist = os.listdir(args.flowFolder)
    flowlist.sort(key=lambda x: int(x[0:-4]))

    # ini_bboxes, end_bboxes, bbox_final_keys=DrawROI(first_rgb_frame,end_rgb_frame)
    # print(ini_bboxes)
    # print(end_bboxes)
    #breakfast2
    # bbox_final_keys=['bowl', 'cereals', 'milk', 'cup']
    # ini_bboxes=[(786, 448, 165, 123), (1104, 243, 145, 186), (987, 366, 171, 241), (1094, 391, 73, 111)]
    # end_bboxes=[(701, 217, 111, 78), (1104, 255, 144, 188), (893, 415, 142, 235), (619, 263, 68, 97)]

    #flower
    bbox_final_keys = ['flower', 'bottle']
    ini_bboxes = [(607, 308, 136, 250), (744, 373, 79, 145)]
    end_bboxes=[(609, 310, 137, 253), (749, 378, 76, 142)]

    # s1_t3_t4
    # bbox_final_keys=['bottle', 'cup']
    # ini_bboxes=[(265, 286, 30, 116), (338, 334, 34, 47)]
    # end_bboxes=[(259, 288, 35, 117), (344, 340, 33, 48)]

    edgebboxes=np.load(args.edgebboxes)['edgebboxes']
    handsbboxes=np.load(args.handsbboxes)['handsbboxes']

    boxes_num=len(ini_bboxes)
    ini_contact_frame=start_contactframe(ini_bboxes,boxes_num,imglist,edgebboxes,handsbboxes)
    end_contact_frame=end_contactframe(end_bboxes,boxes_num,imglist,edgebboxes,handsbboxes)
    print(ini_contact_frame)
    print(end_contact_frame)