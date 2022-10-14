import cv2
import mediapipe as mp
import math
import os
import time



mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0,0,255),thickness=1)
handConStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness=1)
mpHands = mp.solutions.hands
hands=mpHands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.1,min_tracking_confidence=0.7)
pTime=0
cTime=0

def contact_state(box,x,y):
    if box[0]<= x <=(box[0]+box[2]) and box[1]<= y<=(box[1]+box[3]):
        return True
    else:
        return False

def caculate_center(bbox):
    x=bbox[0]+(bbox[2]/2)
    y=bbox[1]+(bbox[3]/2)
    return x,y

def caculate_angle_distance(hc_x,hc_y,oc_x,oc_y):
    dis=math.sqrt((oc_x-hc_x)**2+(oc_y-hc_y)**2)
    a = math.atan2(oc_y-hc_y, oc_x-hc_x)
    # angle = a / math.pi * 180
    return dis,a
# hc_x,hc_y,oc_x,oc_y=0,0,3,4
# print(caculate_angle_distance(hc_x,hc_y,oc_x,oc_y))


def get_startFrames(videofilepath,object_num,object_name_list):
    pTime = 0
    cTime = 0
    dic={}
    dic1={}
    for i in range(len(object_name_list)):
        dic[object_name_list[i]]=[]
        dic1[object_name_list[i]] = []

    print(dic1)
    cap = cv2.VideoCapture(videofilepath)
    ret, frame = cap.read()
    frame_bboxes=[]
    for i in range(object_num):
        cv2.putText(frame, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 1)

        x, y, w, h = cv2.selectROI('display', frame, fromCenter=False)
        init_state = [x, y, w, h]
        frame_bboxes.append(init_state)
    for i in range(object_num):
        dic[object_name_list[i]]=frame_bboxes[i]

    print(dic)
    frame_num=0
    while True:
        ret, frame = cap.read()
        frame_num+=1

        if frame is None:
            break

        height=frame.shape[0]
        width=frame.shape[1]
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result = hands.process(frameRGB)
        for j in dic:
            count=0
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                    # hc_x = handLms.landmark[9].x * width
                    # hc_y = handLms.landmark[9].y * height
                    for i, lm in enumerate(handLms.landmark):
                        xPos = int(lm.x * width)
                        yPos = int(lm.y * height)
                        cv2.putText(frame, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255, 0, 0), 1)
                        if contact_state(dic[j],xPos,yPos):
                            count+=1
            if count>=8:
                dic1[j].append(frame_num)



        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(frame,f"FPS:{int(fps)}",(30,50),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,(255, 0, 0), 1)
        cv2.imshow('display',frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    startframes_list=[]
    for i in dic1:
        print(dic1[i])
        l=[]
        for j in range(len(dic1[i])-1):
            if dic1[i][j+1]-dic1[i][j]==1:
                l.append(dic1[i][j])
            else:
                break
        startframes_list.append(l[-1])
    return startframes_list



videofilepath='/home/zhihao/pytracking/pytracking/videos/reading1_2022-09-08-15-31-12_camera_color_image_raw.mp4'
object_name_list=['glass','news','cup']
object_num=3
print(get_startFrames(videofilepath,object_num,object_name_list))