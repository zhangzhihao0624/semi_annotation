import numpy as np
import collections
from cvzone.HandTrackingModule import HandDetector
import math
import os
import cv2
detector = HandDetector(maxHands=2,detectionCon=0.01,minTrackCon=0.5)

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


def get_edgebboxes(flow_dir):
    flowlist=os.listdir(flow_dir)
    flowlist.sort(key=lambda x: int(x[0:-4]))
    flow_num=len(flowlist)
    edge_bboxes=np.zeros([flow_num,200,4])
    for image_idx in range(flow_num):
        im=cv2.imread(os.path.join(flow_dir,flowlist[image_idx]))
        edgeboxes, score = get_RP_boxes(im, num_boxes=200, p_alpha=0.7, p_beta=0.9, p_eta=1.0, p_min_area=150,gray=False)
        if len(edgeboxes) == 0:
            continue
        elif edgeboxes.shape[0] != 200:
            edge_bboxes[image_idx, 0:edgeboxes.shape[0], :] = edgeboxes
        else:
            edge_bboxes[image_idx, :, :] = edgeboxes


    return edge_bboxes

flow_dir='flower_flow_pro'
edge_bboxes=get_edgebboxes(flow_dir)
np.savez('flower_edgebboxes',edgebboxes=edge_bboxes)

# record=np.load('s1_t3_t4_edgebboxes.npz')
# edgebboxes=record['edgebboxes']
# print(edgebboxes[100,3,:])

def get_handsbboxes(image_dir):
    imagelist = os.listdir(image_dir)
    imagelist.sort(key=lambda x: int(x[0:-4]))
    image_num = len(imagelist)
    handsbboxes = np.zeros([image_num, 2, 4])
    for image_idx in range(image_num):
        frame = cv2.imread(os.path.join(image_dir, imagelist[image_idx]))
        hands = detector.findHands(frame, draw=False)
        hand_num = len(hands)
        if hand_num>2:
            hand_num=2
        for hand_idx in range(hand_num):
            hand = hands[hand_idx]
            hand_bbox = hand['bbox']
            x, y, w, h = hand_bbox
            handsbboxes[image_idx,hand_idx,:]=np.array([x, y, w, h])
    return handsbboxes

image_dir='flower'
handsbboxes=get_handsbboxes(image_dir)
np.savez('flower_handsbboxes',handsbboxes=handsbboxes)
# # print(handsbboxes)
# # print(handsbboxes[100,0,:])
# record=np.load('s1_t3_t4_handsbboxes.npz')
# handsbboxes=record['handsbboxes']
# print(handsbboxes)
imagelist = os.listdir(image_dir)
imagelist.sort(key=lambda x: int(x[0:-4]))
image_num = len(imagelist)
for i in range(image_num-1,-1,-1):
    frame = cv2.imread(os.path.join(image_dir, imagelist[i]))
    for hand_idx in range(2):
        x,y,w,h=handsbboxes[i,hand_idx,:]
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 3)
    cv2.imshow('hands',frame)
    cv2.waitKey(200)
cv2.destroyAllWindows()

