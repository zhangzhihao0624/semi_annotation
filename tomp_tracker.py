import os
import sys
import argparse
import tkinter

import cv2
import numpy as np

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker


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

def runVideo(tracker_name, tracker_param, imagefile,ini_bbox, imglist, optional_box=None, debug=None, save_results=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param)
    bbox=tracker.runVideo(imagefilepath=imagefile, ini_bbox=ini_bbox, imglist=imglist,optional_box=optional_box, debug=debug, save_results=save_results)
    return bbox

def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('--tracker_name', type=str, default='tomp', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='tomp50',help='Name of parameter file.')
    parser.add_argument('--imagefile', type=str, default='/home/zhihao/pytracking/pytracking/imagesfolder/breakfast2/rgb',help='path to a images file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()
    imglist = os.listdir(args.imagefile)
    imglist.sort(key=lambda x: int(x[0:-4]))  # according to the image name choose sorted number

    img_num = len(imglist)
    first_rgb_frame = cv2.imread(os.path.join(args.imagefile, imglist[0]))
    ini_bboxes=[]
    while True:
        cv2.putText(first_rgb_frame, 'Select target ROI and press ENTER', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        cv2.putText(first_rgb_frame, 'press q to quit DrawROI', (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 0, 0), 1)
        x, y, w, h = cv2.selectROI('Initial box', first_rgb_frame, fromCenter=False)
        ini_bboxes.append([x, y, w, h])
        # colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    # ini_bboxes=[[215, 301, 26, 61], [279, 280, 90, 70], [401, 267, 31, 94]]
    # # print(ini_bboxes)
    box_num=len(ini_bboxes)
    bbox_final_keys = InputLabel(box_num)
    print(bbox_final_keys)

    final_bboxes=np.zeros([box_num,img_num,4])
    for box_idx in range(box_num):
        final_bboxes[box_idx,:,:]=runVideo(args.tracker_name, args.tracker_param, args.imagefile,ini_bboxes[box_idx], imglist, optional_box=None, debug=None, save_results=False)
    print(final_bboxes)

    for img_idx in range(img_num):
        im=cv2.imread(os.path.join(args.imagefile, imglist[img_idx]))
        for box_idx in range(box_num):
            bbox=[int(s) for s in final_bboxes[box_idx,img_idx,:]]
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1]),(0, 255, 0), 2)
        cv2.imshow('tracker result', im)
        cv2.waitKey(20)
    cv2.destroyAllWindows()



    bbox_final_dict={}
    for i, key in enumerate(bbox_final_keys):
        bbox_final_dict[key] = final_bboxes[i, :, :]

    print(bbox_final_dict)

    # np.save('breakfast2.npy', bbox_final_dict)




    # print(final_bboxes)

    # run_video(args.tracker_name, args.tracker_param,args.imagefile, args.optional_box, args.debug, args.save_results)


if __name__ == '__main__':
    main()