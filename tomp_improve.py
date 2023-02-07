import os
import sys
import argparse
import tkinter

import cv2
import numpy as np
import torch

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

def CalculateBoxCenter(box):
    return np.array([int(box[0] + box[2]/2), int(box[1] + box[3]/2)])

def is_fix():
    signal = ''


    def GetLabel():
        nonlocal signal

        try:
            signal = var_input.get()

            if signal=='yes' or signal=='no':
                print('signal is accepted')
                root.destroy()
            else:
                var_input.set('The number of labels and drawn boxes must be identical')
        except:
            var_input.set('Input failed, labels should be defined like above')

    root = tkinter.Tk(className='Specify labels')
    root.geometry('600x120')

    show_text = tkinter.Label(root, text="Do you want to fix some objects")
    show_text.pack(pady=20)

    var_input = tkinter.StringVar()
    label_entry = tkinter.Entry(root, textvariable=var_input, width=45)
    label_entry.pack(side='left', expand=True)

    input_button = tkinter.Button(root, text='Input', command=GetLabel)
    input_button.pack(side='left', expand=True, ipadx=20)

    root.mainloop()
    return signal

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

def Inputmethod(obj_name):
    method_name = ''


    def GetLabel():
        nonlocal method_name

        try:
            method_name = var_input.get()

            if method_name=='of' or method_name=='occlusion':
                # for label in label_list:
                #     # str(label.split('_')[0])
                #     # int(label.split('_')[1])
                print('Method input successful')
                root.destroy()
            else:
                var_input.set('The number of labels and drawn boxes must be identical')
        except:
            var_input.set('Input failed, labels should be defined like above')

    root = tkinter.Tk(className='Specify labels')
    root.geometry('600x120')

    show_text = tkinter.Label(root, text="input method to fix " + obj_name +"'s trajectory")
    show_text.pack(pady=20)

    var_input = tkinter.StringVar()
    label_entry = tkinter.Entry(root, textvariable=var_input, width=45)
    label_entry.pack(side='left', expand=True)

    input_button = tkinter.Button(root, text='Input', command=GetLabel)
    input_button.pack(side='left', expand=True, ipadx=20)

    root.mainloop()
    return method_name

def DrawROI(first_rgb_frame, end_rgb_frame):
    ini_bboxes = []
    end_bboxes = []
    colors = []

    while True:
        cv2.putText(first_rgb_frame, 'Select target ROI and press ENTER', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        cv2.putText(first_rgb_frame, 'press q to quit DrawROI', (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        cv2.namedWindow('Initial box', cv2.WND_PROP_FULLSCREEN)
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
        cv2.namedWindow('End box', cv2.WND_PROP_FULLSCREEN)
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

def CalculateHandBoxCenter(box):
    return np.array([int((box[0] + box[2])/2), int((box[1] + box[3])/2)])

def hand_move_trajectory(handsbboxes,frames_interval):
    interval_start=frames_interval[0]
    interval_end=frames_interval[1]
    handscenter=np.zeros([abs(interval_end-interval_start)+1,2,2])
    if interval_start<interval_end:
        n=0
        for i in range(interval_start,interval_end+1):
            for hand_idx in range(2):
                hand_center=CalculateHandBoxCenter(handsbboxes[hand_idx,:,i])
                handscenter[n,hand_idx,:]=hand_center
            n+=1
    else:
        n = 0
        for i in range(interval_start, interval_end - 1,-1):
            for hand_idx in range(2):
                hand_center = CalculateHandBoxCenter(handsbboxes[hand_idx,:,i])
                handscenter[n, hand_idx, :] = hand_center
            n += 1

    return handscenter

def obj_move_trajectory(reference_box,frames_interval,imglist):
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

            frame=cv2.imread(os.path.join(args.imagefile,imglist[frame_idx]))
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
            # img_path = glob.glob(os.path.join(args.imageFolder, str(frame_idx) + '.jpg')) + \
            #            glob.glob(os.path.join(args.imageFolder, str(frame_idx) + '.png'))
            frame=cv2.imread(os.path.join(args.imagefile,imglist[frame_idx]))
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
            hands_boxes=handsbboxes[:,:,image_idx]
            candidate_boxes = []
            # iou filter
            for box in edge_boxes:
                if iou(box, ini_bboxes[box_idx]) > 0.7:
                    candidate_boxes.append([box, iou(box, ini_bboxes[box_idx])])
            candidate_boxes.sort(key=lambda x: x[1], reverse=True)

            if candidate_boxes:
                # hand_preBox_dist=np.zeros([2,1])
                pre_box = candidate_boxes[0][0]
                pre_box_center = CalculateBoxCenter(pre_box)

                for hand_idx in range(2):
                    hand_center=CalculateHandBoxCenter(handsbboxes[hand_idx,:,image_idx])
                    if iou(hands_boxes[hand_idx],pre_box)>0 or np.linalg.norm(pre_box_center-hand_center)<30:
                        start_frames[box_idx].append(image_idx)

    start_tuple=[[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        start_tuple[box_idx]=startframes_tuple(start_frames,box_idx)

    ini_contact=[[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        for tuple_idx in range(len(start_tuple[box_idx])):
            # if len(start_tuple[box_idx][tuple_idx])>=3:
            #     ini_contact[box_idx].append(start_tuple[box_idx][tuple_idx][0])
            # else:
            for i in range(len(start_tuple[box_idx][tuple_idx])):
                frame_interval=[start_tuple[box_idx][tuple_idx][i],start_tuple[box_idx][tuple_idx][i] + 10]
                referencebox=ini_bboxes[box_idx]

                handscenter=hand_move_trajectory(handsbboxes,frame_interval)
                obj_center=obj_move_trajectory(referencebox,frame_interval,imglist)

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
                        if cos_sim>0.9:
                            ini_contact[box_idx].append(frame_interval[0])
                            break

                if len(ini_contact[box_idx])==1:
                    break
            if len(ini_contact[box_idx]) == 1:
                break
    return ini_contact

def end_contactframe(end_bboxes,boxes_num,imglist,edgebboxes,handsbboxes):
    image_num=len(imglist)
    end_frames = [[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        for image_idx in range(image_num-2,-1,-1):
            edge_boxes=edgebboxes[image_idx,:,:]
            hands_boxes=handsbboxes[:,:,image_idx]
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
                    hand_center = CalculateHandBoxCenter(handsbboxes[hand_idx,:,image_idx])
                    if iou(hands_boxes[hand_idx], pre_box) > 0 or np.linalg.norm(pre_box_center - hand_center) < 30:
                        end_frames[box_idx].append(image_idx)
    end_tuple=[[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        end_tuple[box_idx]=endframes_tuple(end_frames,box_idx)

    end_contact = [[] for _ in range(boxes_num)]
    for box_idx in range(boxes_num):
        # if len(end_tuple)==1:
        #     end_contact[box_idx].append(end_tuple[0])
        #     break

        for tuple_idx in range(len(end_tuple[box_idx])):
            # if len(end_tuple[box_idx][tuple_idx])>=3:
            #     end_contact[box_idx].append(end_tuple[box_idx][tuple_idx][0])
            # else:
            for i in range(len(end_tuple[box_idx][tuple_idx])):
                frame_interval = [end_tuple[box_idx][tuple_idx][i], end_tuple[box_idx][tuple_idx][i] - 10]
                referencebox = end_bboxes[box_idx]

                handscenter = hand_move_trajectory(handsbboxes, frame_interval)
                obj_center = obj_move_trajectory(referencebox, frame_interval,imglist)

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
                        break
                    if not (np.any(v2)):
                        continue
                    else:
                        cos_sim = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        if cos_sim > 0.8:
                            end_contact[box_idx].append(frame_interval[0])
                            break

                if len(end_contact[box_idx]) == 1:
                    break
            if len(end_contact[box_idx]) == 1:
                break
    return end_contact

# def start_contactframe(ini_bboxes,boxes_num,imglist,edgebboxes,handsbboxes):
#     image_num=len(imglist)
#     start_frames = [[] for _ in range(boxes_num)]
#     for box_idx in range(boxes_num):
#         for image_idx in range(image_num-1):
#             edge_boxes=edgebboxes[image_idx,:,:]
#             hands_boxes=handsbboxes[image_idx,:,:]
#             candidate_boxes = []
#             # iou filter
#             for box in edge_boxes:
#                 if iou(box, ini_bboxes[box_idx]) > 0.5:
#                     candidate_boxes.append([box, iou(box, ini_bboxes[box_idx])])
#             candidate_boxes.sort(key=lambda x: x[1], reverse=True)
#
#             if candidate_boxes:
#                 # hand_preBox_dist=np.zeros([2,1])
#                 pre_box = candidate_boxes[0][0]
#                 pre_box_center = CalculateBoxCenter(pre_box)
#
#                 # for hand_idx in range(2):
#                 #     hand_center=CalculateBoxCenter(hands_boxes[hand_idx])
#                 #     if iou(hands_boxes[hand_idx],pre_box)>0 or np.linalg.norm(pre_box_center-hand_center)<40:
#                 start_frames[box_idx].append(image_idx)
#
#     start_tuple=[[] for _ in range(boxes_num)]
#     for box_idx in range(boxes_num):
#         start_tuple[box_idx]=startframes_tuple(start_frames,box_idx)
#
#     ini_contact=[[] for _ in range(boxes_num)]
#     for box_idx in range(boxes_num):
#         for tuple_idx in range(len(start_tuple[box_idx])):
#             if len(start_tuple[box_idx][tuple_idx])>=3:
#                 ini_contact[box_idx].append(start_tuple[box_idx][tuple_idx][0])
#             else:
#                 for i in range(len(start_tuple[box_idx][tuple_idx])):
#                     frame_interval=[start_tuple[box_idx][tuple_idx][i],start_tuple[box_idx][tuple_idx][i] + 5]
#                     referencebox=ini_bboxes[box_idx]
#
#                     handscenter=hand_move_trajectory(handsbboxes,frame_interval)
#                     obj_center=obj_move_trajectory(referencebox,frame_interval,imglist)
#
#                     obj_movingvector = np.diff(obj_center, axis=0)
#                     average_objmovingvector = np.mean(obj_movingvector, axis=0)
#                     average_handmovingvector = np.zeros([2, 2])
#                     for hand_idx in range(2):
#                         hand_moingvector = np.diff(handscenter[:, hand_idx, :], axis=0)
#                         average_handmovingvector[hand_idx, :] = np.mean(hand_moingvector, axis=0)
#
#                     for hand_idx in range(2):
#                         v1 = average_objmovingvector
#                         v2 = average_handmovingvector[hand_idx]
#                         if np.linalg.norm(v1)<1:
#                             continue
#                         if not(np.any(v2)):
#                             continue
#                         else:
#                             cos_sim = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#                             if cos_sim>0.8:
#                                 ini_contact[box_idx].append(frame_interval[0])
#                                 break
#
#                     if len(ini_contact[box_idx])==1:
#                         break
#             if len(ini_contact[box_idx]) == 1:
#                 break
#     return ini_contact
#
# def end_contactframe(end_bboxes,boxes_num,imglist,edgebboxes,handsbboxes):
#     image_num=len(imglist)
#     end_frames = [[] for _ in range(boxes_num)]
#     for box_idx in range(boxes_num):
#         for image_idx in range(image_num-2,-1,-1):
#             edge_boxes=edgebboxes[image_idx,:,:]
#             hands_boxes=handsbboxes[image_idx,:,:]
#             candidate_boxes = []
#             # iou filter
#             for box in edge_boxes:
#                 if iou(box, end_bboxes[box_idx]) > 0.5:
#                     candidate_boxes.append([box, iou(box, end_bboxes[box_idx])])
#             candidate_boxes.sort(key=lambda x: x[1], reverse=True)
#
#             if candidate_boxes:
#                 # hand_preBox_dist=np.zeros([2,1])
#                 pre_box = candidate_boxes[0][0]
#                 pre_box_center = CalculateBoxCenter(pre_box)
#                 # for hand_idx in range(2):
#                 #     hand_center = CalculateBoxCenter(hands_boxes[hand_idx])
#                 #     if iou(hands_boxes[hand_idx], pre_box) > 0 or np.linalg.norm(pre_box_center - hand_center) < 50:
#                 end_frames[box_idx].append(image_idx)
#     end_tuple=[[] for _ in range(boxes_num)]
#     for box_idx in range(boxes_num):
#         end_tuple[box_idx]=endframes_tuple(end_frames,box_idx)
#
#     end_contact = [[] for _ in range(boxes_num)]
#     for box_idx in range(boxes_num):
#         for tuple_idx in range(len(end_tuple[box_idx])):
#             if len(end_tuple[box_idx][tuple_idx])>=3:
#                 end_contact[box_idx].append(end_tuple[box_idx][tuple_idx][0])
#             else:
#                 for i in range(len(end_tuple[box_idx][tuple_idx])):
#                     frame_interval = [end_tuple[box_idx][tuple_idx][i], end_tuple[box_idx][tuple_idx][i] - 10]
#                     referencebox = end_bboxes[box_idx]
#
#                     handscenter = hand_move_trajectory(handsbboxes, frame_interval)
#                     obj_center = obj_move_trajectory(referencebox, frame_interval,imglist)
#
#                     obj_movingvector = np.diff(obj_center, axis=0)
#                     average_objmovingvector = np.mean(obj_movingvector, axis=0)
#                     average_handmovingvector = np.zeros([2, 2])
#                     for hand_idx in range(2):
#                         hand_moingvector = np.diff(handscenter[:, hand_idx, :], axis=0)
#                         average_handmovingvector[hand_idx, :] = np.mean(hand_moingvector, axis=0)
#
#                     for hand_idx in range(2):
#                         v1 = average_objmovingvector
#                         v2 = average_handmovingvector[hand_idx]
#                         if np.linalg.norm(v1) < 1:
#                             break
#                         if not (np.any(v2)):
#                             continue
#                         else:
#                             cos_sim = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#                             if cos_sim > 0.8:
#                                 end_contact[box_idx].append(frame_interval[0])
#                                 break
#
#                     if len(end_contact[box_idx]) == 1:
#                         break
#             if len(end_contact[box_idx]) == 1:
#                 break
#     return end_contact

def of_siamtracker(ini_box,tracker_interval,flowlist):
    track_start=tracker_interval[0]
    track_end=tracker_interval[1]
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
    tracker_boxes = np.zeros([4,abs(track_end - track_start)])
    n = 0
    for frame_idx in range(track_start, track_end):
        frame = cv2.imread(os.path.join(args.flowFolder, flowlist[frame_idx]))
        # frame = cv2.imread(os.path.join(args.imageFolder, 'frame_' + str(frame_idx) + '.png'))
        if first_frame:
            init_rect = ini_box
            tracker.init(frame, init_rect)
            x1 = ini_box[0]
            y1 = ini_box[1]
            x2 = ini_box[0] + ini_box[2]
            y2 = ini_box[1] + ini_box[3]
            tracker_boxes[:,n] = np.array([x1, y1, x2, y2])
            first_frame = False
        else:
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[0] + bbox[2]
            y2 = bbox[1] + bbox[3]
            tracker_boxes[:,n]=np.array([x1, y1, x2, y2])

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
            cv2.waitKey(20)
        n += 1
    cv2.destroyAllWindows()

    return tracker_boxes

def occlusion_solve(end_box,tracker_interval,imglist):
    # track_start = tracker_interval[1]
    # track_end = tracker_interval[0]


    # tracker_boxes = np.zeros([4, abs(track_end - track_start)])
    tracker_boxes=tomp(args.tracker_name, args.tracker_param, tracker_interval,args.imagefile,end_box, imglist)

    return tracker_boxes


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


def runVideo(tracker_name, tracker_param, imagefile,ini_bbox, imglist, optional_box=None, debug=None, save_results=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param)
    bbox=tracker.runVideo(imagefilepath=imagefile, ini_bbox=ini_bbox, imglist=imglist,optional_box=optional_box, debug=debug)
    return bbox

def tomp(tracker_name, tracker_param, tracker_interval,imagefile,ini_bbox, imglist):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param)
    bbox=tracker.tomp_tracker(ini_bbox,tracker_interval=tracker_interval,imagefilepath=imagefile,imglist=imglist, optional_box=None, debug=None, visdom_info=None)
    return bbox





if __name__ == '__main__':
    scene_folder = "makeup_t1"

    current_path = os.getcwd()
    scene_path = os.path.join(current_path, scene_folder)
    rgb_folder = os.path.join(current_path, scene_folder, "rgb")
    flow_folder= os.path.join(current_path,scene_folder ,"flow")
    record_file = scene_folder + ".npz"
    edgebboxes_file = os.path.join(current_path,scene_folder , "edgebboxes.npz")
    # handsbboxes_file= os.path.join(current_path,scene_folder , "handsbboxes.npz")
    GT_dir = os.path.join(current_path, scene_folder, "obj_train_data")

    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('--tracker_name', type=str, default='tomp', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='tomp50', help='Name of parameter file.')
    parser.add_argument('--config', type=str,
                        default='/home/zhihao/pysot-master/experiments/siamrpn_r50_l234_dwxcorr/config.yaml',
                        help='config file')
    parser.add_argument('--snapshot', type=str,
                        default='/home/zhihao/pysot-master/experiments/siamrpn_r50_l234_dwxcorr/model.pth')
    parser.add_argument('--imagefile', type=str, default=rgb_folder,
                        help='path to a images file.')
    parser.add_argument('--flowFolder', default=flow_folder, type=str,
                        help='Raft flow files')
    parser.add_argument('--edgebboxes', default=edgebboxes_file,
                        type=str,
                        help='Raft flow files')
    # parser.add_argument('--handsbboxes', default=handsbboxes_file,
    #                     type=str,
    #                     help='Raft flow files')
    # parser.add_argument('--GTdir', default=GT_dir,
    #                     type=str, help='Groundtruth files')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()
    colors = {0: [0, 0, 255], 1: [0, 255, 0], 2: [0, 255, 255], 3: [255, 0, 0], 4: [139, 72, 61],
              5: [209, 206, 0], 6: [224, 255, 255]}
    imglist = os.listdir(args.imagefile)
    imglist.sort(key=lambda x: int(x[0:-4]))  # according to the image name choose sorted number

    flowlist = os.listdir(args.flowFolder)
    flowlist.sort(key=lambda x: int(x[0:-4]))

    record = np.load(os.path.join(scene_path, record_file))
    hand_record = record["hand_record"]

    edgebboxes = np.load(args.edgebboxes)['edgebboxes']
    handsbboxes=hand_record[:,0:4,:]

    img_num = len(imglist)
    first_rgb_frame = cv2.imread(os.path.join(args.imagefile, imglist[0]))
    end_rgb_frame = cv2.imread(os.path.join(args.imagefile, imglist[-1]))
    image_size = first_rgb_frame.shape[:2][::-1]
    ini_bboxes = []
    while True:
        cv2.putText(first_rgb_frame, 'Select target ROI and press ENTER', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        cv2.putText(first_rgb_frame, 'press q to quit DrawROI', (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        cv2.namedWindow('Initial box', cv2.WND_PROP_FULLSCREEN)
        x, y, w, h = cv2.selectROI('Initial box', first_rgb_frame, fromCenter=False)
        ini_bboxes.append([x, y, w, h])
        # colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    # ini_bboxes=[[215, 301, 26, 61], [279, 280, 90, 70], [401, 267, 31, 94]]
    # # print(ini_bboxes)
    box_num = len(ini_bboxes)
    bbox_final_keys = InputLabel(box_num)
    print(bbox_final_keys)

    final_bboxes = np.zeros([box_num, 4, img_num])
    # for box_idx in range(box_num):
    #     final_bboxes[box_idx, :, :] = runVideo(args.tracker_name, args.tracker_param, args.imagefile,
    #                                            ini_bboxes[box_idx], imglist, optional_box=None, debug=None,
    #                                            save_results=False)
    # print(final_bboxes)

    for box_idx in range(box_num):
        box = ini_bboxes[box_idx]
        x1=box[0]
        y1=box[1]
        x2=box[0]+box[2]
        y2=box[1]+box[3]
        final_bboxes[box_idx,:,0]=np.array([x1, y1, x2, y2])
    bbox=runVideo(args.tracker_name, args.tracker_param, args.imagefile,ini_bboxes, imglist, optional_box=None, debug=None, save_results=False)
    final_bboxes[:,:,1:]=bbox[:,:,1:]

    # for img_idx in range(img_num):
    #     im=cv2.imread(os.path.join(args.imagefile, imglist[img_idx]))
    #     for box_idx in range(box_num):
    #         bbox=[int(s) for s in final_bboxes[box_idx,:,img_idx]]
    #         cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2] , bbox[3]),(0, 255, 0), 2)
    #     cv2.imshow('tracker result', im)
    #     cv2.waitKey(10)
    # cv2.destroyAllWindows()

    # bbox_final_dict = np.load('/home/zhihao/pytracking/pytracking/demo_test.npy', allow_pickle=True).item()
    bbox_final_dict = {}
    for i, key in enumerate(bbox_final_keys):
        bbox_final_dict[key] = final_bboxes[i, :, :]

    # colors = {'bowl': [0, 255, 255], 'cereal': [0, 0, 255], 'milk': [0, 255, 0], 'cup': [255, 0, 0]}
    # for image_idx in range(img_num):
    #     im=cv2.imread(os.path.join(args.imagefile, imglist[image_idx]))
    #     for key in bbox_final_keys:
    #         x1, y1, x2, y2 = map(int, bbox_final_dict[key][:, image_idx])
    #         cv2.rectangle(im, (x1, y1), (x2, y2), colors[key], 2, cv2.LINE_AA)
    #         cv2.putText(im, key, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[key], 1)
    #     cv2.imshow('tomp tracker ', im)
    #     cv2.waitKey(50)
    # cv2.destroyAllWindows()

    signal=is_fix()

    if signal=='no':
        # bbox_final_dict = {}
        # for i, key in enumerate(bbox_final_keys):
        #     bbox_final_dict[key] = final_bboxes[i, :, :]

        GT_dict = load_groundtruth(args.GTdir, image_size, box_num, bbox_final_keys)

        if len(bbox_final_dict) == len(GT_dict):
            accuracy_dict, average_iou, TP, FP, FN = CalculateIOU(bbox_final_dict, GT_dict)
            print('TP: {}, FP: {}, FN:{}'.format(TP, FP, FN))
            print('Precision: {}, Recall: {}'.format(TP / (TP + FP), TP / (TP + FN)))
            print('accuracy: {}'.format(accuracy_dict))
            print('average iou: {}'.format(average_iou))

        else:
            raise Exception('number of ground truth labels is not identical to number of drawn bounding boxes')
    else:
        first_bboxes, last_bboxes, fix_bbox_keys = DrawROI(first_rgb_frame, end_rgb_frame)
        fix_boxes_num = len(first_bboxes)
        ini_contact_frame = start_contactframe(first_bboxes, fix_boxes_num, imglist, edgebboxes, handsbboxes)
        end_contact_frame = end_contactframe(last_bboxes, fix_boxes_num, imglist, edgebboxes, handsbboxes)
        print(ini_contact_frame)
        print(end_contact_frame)

        contact_stage = [[] for _ in range(fix_boxes_num)]
        for box_idx in range(fix_boxes_num):
            contact_stage[box_idx].append(ini_contact_frame[box_idx][0])
            contact_stage[box_idx].append(end_contact_frame[box_idx][0])
        print(contact_stage)
        # contact_stage=[[73,404]]
        fix_bboxes = np.zeros([fix_boxes_num, 4, img_num])
        for box_idx in range(fix_boxes_num):
            fix_method = Inputmethod(fix_bbox_keys[box_idx])
            for img_idx in range(contact_stage[box_idx][0]):
                fix_bboxes[box_idx,:,img_idx] = [first_bboxes[box_idx][0],first_bboxes[box_idx][1],first_bboxes[box_idx][0]+first_bboxes[box_idx][2],first_bboxes[box_idx][1]+first_bboxes[box_idx][3]]
            for img_idx in range(contact_stage[box_idx][1], img_num):
                fix_bboxes[box_idx,:,img_idx] = [last_bboxes[box_idx][0],last_bboxes[box_idx][1],last_bboxes[box_idx][0]+last_bboxes[box_idx][2],last_bboxes[box_idx][1]+last_bboxes[box_idx][3]]
            if fix_method == 'of':
                fix_bboxes[box_idx, :, contact_stage[box_idx][0]:contact_stage[box_idx][1]] = of_siamtracker(
                    first_bboxes[box_idx], contact_stage[box_idx], flowlist)
            elif fix_method == 'occlusion':
                fix_bboxes[box_idx, :, contact_stage[box_idx][0]:contact_stage[box_idx][1]] = tomp(args.tracker_name, args.tracker_param, contact_stage[box_idx],args.imagefile,
                                                                                                   last_bboxes[box_idx], imglist)

        for i, key in enumerate(fix_bbox_keys):
            # bbox_final_dict[key][:,contact_stage[i][0]:contact_stage[i][1]] = fix_bboxes[i, :, contact_stage[i][0]:contact_stage[i][1]]
            bbox_final_dict[key][:, :] = fix_bboxes[i, :, :]

        # for image_idx in range(img_num):
        #     im = cv2.imread(os.path.join(args.imagefile, imglist[image_idx]))
        #     for key in bbox_final_keys:
        #         x1, y1, x2, y2 = map(int, bbox_final_dict[key][:, image_idx])
        #         cv2.rectangle(im, (x1, y1), (x2, y2), colors[key], 2, cv2.LINE_AA)
        #         cv2.putText(im, key, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[key], 1)
        #     cv2.imshow('final result ', im)
        #     cv2.waitKey(30)
        # cv2.destroyAllWindows()

        GT_dict = load_groundtruth(args.GTdir, image_size, box_num, bbox_final_keys)

        if len(bbox_final_dict) == len(GT_dict):
            accuracy_dict, average_iou, TP, FP, FN = CalculateIOU(bbox_final_dict, GT_dict)
            print('TP: {}, FP: {}, FN:{}'.format(TP, FP, FN))
            print('Precision: {}, Recall: {}'.format(TP / (TP + FP), TP / (TP + FN)))
            print('accuracy: {}'.format(accuracy_dict))
            print('average iou: {}'.format(average_iou))

        else:
            raise Exception('number of ground truth labels is not identical to number of drawn bounding boxes')











    # bbox_final_dict = {}
    # for i, key in enumerate(bbox_final_keys):
    #     bbox_final_dict[key] = final_bboxes[i, :, :]
    #
    # GT_dict = load_groundtruth(args.GTdir, image_size, box_num, bbox_final_keys)
    #
    # if len(bbox_final_dict) == len(GT_dict):
    #     accuracy_dict, average_iou, TP, FP, FN = CalculateIOU(bbox_final_dict, GT_dict)
    #     print('TP: {}, FP: {}, FN:{}'.format(TP, FP, FN))
    #     print('Precision: {}, Recall: {}'.format(TP / (TP + FP), TP / (TP + FN)))
    #     print('accuracy: {}'.format(accuracy_dict))
    #     print('average iou: {}'.format(average_iou))
    #
    # else:
    #     raise Exception('number of ground truth labels is not identical to number of drawn bounding boxes')