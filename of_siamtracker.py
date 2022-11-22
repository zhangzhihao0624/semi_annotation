'''''
根据光流和接触开始结束帧标注整个视频
'''''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)
''''
输入tracker模型参数，以及要处理的光流文件
'''''
parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str,default='/home/zhihao/pysot-master/experiments/siamrpn_r50_l234_dwxcorr/config.yaml' ,help='config file')
parser.add_argument('--snapshot', type=str, default='/home/zhihao/pysot-master/experiments/siamrpn_r50_l234_dwxcorr/model.pth')
parser.add_argument('--video_name', default='/home/zhihao/pysot-master/demo/test_result', type=str,
                    help='videos or image files')
args = parser.parse_args()

def iou(a,b):

    area_a = a[2] * a[3]
    area_b = b[2] * b[3]

    w = min(b[0]+b[2],a[0]+a[2]) - max(a[0],b[0])
    h = min(b[1]+b[3],a[1]+a[3]) - max(a[1],b[1])

    if w <= 0 or h <= 0:
        return 0

    area_c = w * h

    return area_c / (area_a + area_b - area_c)


images = glob(os.path.join(args.video_name, '*.png'))
images = sorted(images,
                key=lambda x: int(x.split('/')[-1][0:-4]))

# for img in images:
#     frame = cv2.imread(img)
#     cv2.imshow('f',frame)
#     cv2.waitKey(1)
# cv2.destroyAllWindows()
f=open('/home/zhihao/pysot-master/flower1.txt')
frames_num=int(f.readline())
objects_num=int(f.readline())
objects_name=eval(f.readline())
first_boxes=eval(f.readline())
last_boxes=eval(f.readline())
start_Contactframes=eval(f.readline())
end_Contactframes=eval(f.readline())
# print(start_Contactframes)
# print(end_Contactframes)
f.close()
bboxs=np.zeros((frames_num,objects_num,4))
for j in range(objects_num):
    for i in range(start_Contactframes[objects_name[j]]+1):
        bboxs[i][j]=first_boxes[objects_name[j]]
    for k in range(end_Contactframes[objects_name[j]],frames_num):
        bboxs[k][j]=last_boxes[objects_name[j]]
# print(bboxs)
a=start_Contactframes[objects_name[0]]
b=end_Contactframes[objects_name[0]]
# for img in images[a:b]:
#     frame = cv2.imread(img)
#     # yield frame
#     cv2.imshow('f',frame)
#     cv2.waitKey(1)
# cv2.destroyAllWindows()

def get_frames(images,a,b):
    # images = glob(os.path.join(video_name, '*.png'))
    # images = sorted(images,
    #                 key=lambda x: int(x.split('/')[-1][0:-4]))
    for img in images[a:b]:
        frame = cv2.imread(img)
        yield frame

# for frame in get_frames(image_list)
def main():
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

    for k in range(objects_num):
        tmp=[]
        a = start_Contactframes[objects_name[k]]
        b = end_Contactframes[objects_name[k]]

        first_frame = True
        if args.video_name:
            video_name = args.video_name.split('/')[-1].split('.')[0]
        else:
            video_name = 'webcam'
        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
        n=a
        for frame in get_frames(images,a,b):
            n+=1
            if first_frame:
                try:
                    init_rect = first_boxes[objects_name[k]]
                    tmp.append(init_rect)
                except:
                    exit()
                tracker.init(frame, init_rect)
                first_frame = False
            else:
                outputs = tracker.track(frame)
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
                    # if iou(bbox,tmp[-1])>0.3:
                    #     bboxs[n][k]=bbox
                    # else:
                    #     bboxs[n][k]=tmp[-1]

                    tmp.append(bbox)
                    bboxs[n][k] = bbox
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 3)
                cv2.imshow(video_name, frame)
                cv2.waitKey(1)


if __name__ == '__main__':
    main()
    images = glob(os.path.join('/home/zhihao/pysot-master/demo/flower', '*.jpg'))
    images = sorted(images,
                    key=lambda x: int(x.split('/')[-1][0:-4]))
    # print(images)
    for i in range(frames_num):
        im=cv2.imread(images[i])
        for k in range(objects_num):
            for bb in bboxs[i]:
                # bb=b.tolist()
                # print(bb,type(bb))
                x=int(bb[0])
                y=int(bb[1])
                w=int(bb[2])
                h=int(bb[3])
                cv2.rectangle(im, (x, y),
                              (x + w, y + h),
                              (0, 255, 0), 3)

        cv2.imshow('test', im)
        cv2.waitKey(20)
