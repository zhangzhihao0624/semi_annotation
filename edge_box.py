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


#  yolo format to coco [x_min,y_min,w,h]
def yolo2coco(bboxes, image_height=480, image_width=640):
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

def get_RP_boxes(im,num_boxes=200,p_alpha=0.7,p_beta=0.9,p_eta=1.0,p_min_area=150,gray=False):
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('/home/zzh/RAFT/model.yml')
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

    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    # for b in boxes[0]:
    #     x, y, w, h = b
    #     cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    return(im,edges,orimap,boxes)

def get_GT_bboxes(GT_path):
    # GT_path='/home/zzh/RAFT/GT_2d/frame_0.txt'
    with open(GT_path, 'r') as f:
        data = f.read()
        data = data.split()
        data = list(map(float, data))
    object_num = int(len(data) / 5)
    bbox_list = []
    for i in range(object_num):
        bbox_list.append([data[1 + i * 5], data[2 + i * 5], data[3 + i * 5], data[4 + i * 5]])

    bbox_list = np.array(bbox_list)
    GT = []
    for i in range(len(bbox_list)):
        GT.append(yolo2coco(bbox_list[i],image_height=480, image_width=640))

    GT = np.array(GT)
    return(GT)


GT_path = '/home/zzh/RAFT/GT_2d/frame_314.txt'
GT = get_GT_bboxes(GT_path)
# print(GT)


# -------------get edge and proposal_boxes--------------
im=cv2.imread('optical-flow/314.png')
im,edges,orimap,boxes=get_RP_boxes(im,num_boxes=200,p_alpha=0.7,p_beta=0.9,p_eta=1.0,p_min_area=150,gray=False)
# # print(boxes[0])


# ----------show edge boxes----------
# for b in boxes[0]:
#     x, y, w, h = b
#     cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
# cv2.imshow('flow',im)
# cv2.waitKey(0)

# ---------iou filter--------------
# count=0
for i in range(len(GT)):
    for b in boxes[0]:
        x, y, w, h = b
        if iou(GT[i],b)>0.9:
            GT[i]=b
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
cv2.imshow('flow',im)
cv2.waitKey(0)
# # print(count)





# -----------------------function -----------------------


def egde_boxes_tracker(flow_folder,GT_path):
    flowlist = os.listdir(flow_folder)
    flowlist.sort(key=lambda x: int(x[0:-4]))
    flow_num=len(flowlist)
    # print(flowlist)
    # print(flow_num)
    GT = get_GT_bboxes(GT_path)
    new_boxes=np.zeros((421,10,4))
    # print(new_boxes[0,:,:])
    # print(new_boxes[1,:,:].all()==0)
    # print(new_boxes.shape)
    # print(new_boxes)
    for i in range(flow_num):
        im = cv2.imread(os.path.join(flow_folder,flowlist[i]))
        im, edges, orimap, boxes = get_RP_boxes(im, num_boxes=200, p_alpha=0.7, p_beta=0.9, p_eta=1.0, p_min_area=150,gray=False)
        cv2.imshow('flow',im)
        cv2.waitKey(10)
        for j in range(len(GT)):
            for b in boxes[0]:
                x, y, w, h = b
                if iou(GT[j],b)>0.7:
                    new_boxes[i,:,:][0]=b
                    GT[j]=b



        # elif i>1:
        #     if new_boxes[i-1,:,:][0].any()>0:
        #         a=new_boxes[i-1,:,:][0]
        #         for b in boxes[0]:
        #             x, y, w, h = b
        #             if iou(a,b) > 0.75:
        #                 new_boxes[i,:,:]=b
        #
        #
        #     else:
        #         for a in GT:
        #             for b in boxes[0]:
        #                 x, y, w, h = b
        #                 if iou(a, b) > 0.7:
        #                     new_boxes[i, :, :] = b

        print(new_boxes[i,:,:][0])

        if new_boxes[i,:,:][0].any()>0:
            b=new_boxes[i,:,:][0]
            x=int(b[0])
            y=int(b[1])
            w=int(b[2])
            h=int(b[3])
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('bbox',im)
            cv2.waitKey(10)

#
# flow_folder='optical-flow'
# GT_path = '/home/zzh/RAFT/GT_2d/frame_30.txt'
# egde_boxes_tracker(flow_folder,GT_path)


