import cv2
import numpy as np
import os

#iou filter,output iou,between (0-1)
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

    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    # for b in boxes[0]:
    #     x, y, w, h = b
    #     cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    return(im,edges,orimap,boxes)



def contactStart_frames(first_boxes,flow_folder):
    flowlist = os.listdir(flow_folder)
    flowlist.sort(key=lambda x: int(x[0:-4]))
    flow_num=len(flowlist)
    dic1={}
    obj_start_frame={}
    for l in range(flow_num):
        im = cv2.imread(os.path.join(flow_folder, flowlist[l]))
        # produce proposal bounding boxes according to the edge features
        im, edges, orimap, boxes = get_RP_boxes(im, num_boxes=200, p_alpha=0.7, p_beta=0.9, p_eta=1.0, p_min_area=150,
                                                gray=False)
        candidate_boxes = []
        for k in first_boxes:           # iou filter
            for b in boxes[0]:
                if iou(first_boxes[k],b)>0.5:
                    candidate_boxes.append(b)
        candidate_boxes_num = len(candidate_boxes)
        w = []
        if candidate_boxes_num == 1:  # 如果candidate boxes只有一个，那么作为预测框输出
            w.append(candidate_boxes[0])
        elif candidate_boxes_num > 1:  # 如果candidate boxes有多个，candidate boxes之间进行iou filter，保证输出一个最可能的预测框
            for i in range(candidate_boxes_num):
                z = []
                for j in range(candidate_boxes_num):
                    if i != j:
                        z.append(iou(candidate_boxes[i], candidate_boxes[j]))
                w.append(sum(z) / len(z))
        else:  # 没有满足的 candidate boxes, 没有预测框输出
            w = []

        if w:                                  #产生了预测框的情况下，把新的预测框跟GT进行iou比较，满足条件的就把预测框替换原本的GT，实现物体边界框的更新
            max_index = w.index(max(w))        #平均IOU最大的candidate boxes为OUTPUT
            for z in first_boxes:
                if iou(first_boxes[z], candidate_boxes[max_index]) > 0.7:
                    if z not in obj_start_frame:
                        dic1[z] = candidate_boxes[max_index]
                        obj_start_frame[z]=l+1
                    else:  # 已经存在，比较哪一个IOU更接近GT，选择更接近的作为输出
                        if iou(candidate_boxes[max_index], first_boxes[z]) > iou(first_boxes[z], dic1[z]):
                            obj_start_frame[z] = l+1
                            dic1[z] = candidate_boxes[max_index]

            x=int(candidate_boxes[max_index][0])
            y=int(candidate_boxes[max_index][1])
            w=int(candidate_boxes[max_index][2])
            h=int(candidate_boxes[max_index][3])
            # colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],[0, 0, 0],
            #           [255, 0, 255], [0, 0, 0], [100, 40, 50]]
            cv2.rectangle(im, (x, y), (x + w, y + h), [0,255,0], 1, cv2.LINE_AA)
            # flow_path=os.path.join('s1_t5_t4_tracking_example',str(l)+'.png')     #保存画完边界框的图片
            # cv2.imwrite(flow_path, im)
            cv2.imshow('frame', im)
            cv2.waitKey(1)
        else:
            cv2.imshow('frame' , im)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    return obj_start_frame


def contactEnd_frames(last_boxes,flow_folder):
    flowlist = os.listdir(flow_folder)
    flowlist.sort(key=lambda x: int(x[0:-4]),reverse=True)
    flow_num=len(flowlist)
    dic1={}

    obj_end_frame={}
    for l in range(flow_num):
        im = cv2.imread(os.path.join(flow_folder, flowlist[l]))
        # produce proposal bounding boxes according to the edge features
        im, edges, orimap, boxes = get_RP_boxes(im, num_boxes=200, p_alpha=0.7, p_beta=0.9, p_eta=1.0, p_min_area=150,
                                                gray=False)
        candidate_boxes = []
        for k in last_boxes:
            for b in boxes[0]:
                if iou(last_boxes[k],b)>0.5:
                    candidate_boxes.append(b)
        candidate_boxes_num = len(candidate_boxes)
        w = []
        if candidate_boxes_num == 1:  # 如果candidate boxes只有一个，那么作为预测框输出
            w.append(candidate_boxes[0])
        elif candidate_boxes_num > 1:  # 如果candidate boxes有多个，candidate boxes之间进行iou filter，保证输出一个最可能的预测框
            for i in range(candidate_boxes_num):
                z = []
                for j in range(candidate_boxes_num):
                    if i != j:
                        z.append(iou(candidate_boxes[i], candidate_boxes[j]))
                w.append(sum(z) / len(z))
        else:  # 没有满足的 candidate boxes, 没有预测框输出
            w = []

        if w:                                  #产生了预测框的情况下，把新的预测框跟GT进行iou比较，满足条件的就把预测框替换原本的GT，实现物体边界框的更新
            max_index = w.index(max(w))        #平均IOU最大的candidate boxes为OUTPUT
            for z in last_boxes:
                if iou(last_boxes[z], candidate_boxes[max_index]) > 0.7:
                    if z not in obj_end_frame:
                        dic1[z]=candidate_boxes[max_index]         #保存这个BBOX
                        obj_end_frame[z]=flow_num-l-1
                    else:                                           #已经存在，比较哪一个IOU更接近GT，选择更接近的作为输出
                        if iou(candidate_boxes[max_index],last_boxes[z])>iou(last_boxes[z],dic1[z]):
                            obj_end_frame[z] = flow_num - l-1
                            dic1[z] = candidate_boxes[max_index]


            x=int(candidate_boxes[max_index][0])
            y=int(candidate_boxes[max_index][1])
            w=int(candidate_boxes[max_index][2])
            h=int(candidate_boxes[max_index][3])

            cv2.rectangle(im, (x, y), (x + w, y + h), [0,255,0], 1, cv2.LINE_AA)
            # flow_path=os.path.join('s1_t5_t4_tracking_example',str(l)+'.png')     #保存画完边界框的图片
            # cv2.imwrite(flow_path, im)
            cv2.imshow('frame', im)
            cv2.waitKey(1)
        else:
            cv2.imshow('frame' , im)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    return obj_end_frame,flow_num


#--------------------input first frame and last frame,draw bbox------------
first_frame='makeup/1.jpg'
end_frame='makeup/979.jpg'
#--------------------input optical flow folder------------------------------
flow_folder='makeup_flow_30'
#--------------------input objects number and give objects name------------
objects_num=4
objects_name=['mirror','perfume','comb','lipstick']
#---------------------manual draw first and last frame bbox---------------
first_boxes={}
last_boxes={}
image1=cv2.imread(first_frame)
for i in range(objects_num):
    cv2.putText(image1, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
               1.5, (0, 0, 0), 1)

    x, y, w, h = cv2.selectROI('first frame', image1, fromCenter=False)
    init_state = [x, y, w, h]
    first_boxes[objects_name[i]]=init_state
print(first_boxes)
image2=cv2.imread(end_frame)
for i in range(objects_num):
    cv2.putText(image2, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
               1.5, (0, 0, 0), 1)

    x, y, w, h = cv2.selectROI('last frame', image2, fromCenter=False)
    init_state = [x, y, w, h]
    last_boxes[objects_name[i]]=init_state
print(last_boxes)

# print('objects contact start frames',contactStart_frames(first_boxes,flow_folder))
# print('objects contact end frames',contactEnd_frames(last_boxes,flow_folder))
start_Contactframes=contactStart_frames(first_boxes,flow_folder)
end_Contactframes,flow_num=contactEnd_frames(last_boxes,flow_folder)
#--------------------------save results as txt file--------------------------------
file=open("makeup.txt", 'w')
file.writelines(str(flow_num+1)+'\n')         #frame number
file.writelines(str(objects_num)+'\n')        #objects number
file.writelines(str(objects_name)+'\n')       #objects name list
file.writelines(str(first_boxes)+'\n')        #manual annotate bbox in first frame
file.writelines(str(last_boxes)+'\n')         #manual annotate bbox in last frame
file.writelines(str(start_Contactframes)+'\n') #+
file.writelines(str(end_Contactframes)+'\n')
file.close()