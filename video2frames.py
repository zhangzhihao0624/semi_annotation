import cv2

# load video from your path
vc = cv2.VideoCapture('/home/zhihao/RAFT/TUM_dataset/reading_t2.mp4')   #导入视频地址
c = 0
rval = vc.isOpened()

while rval:

    rval, frame = vc.read()
    if rval:
        cv2.imwrite('/home/zhihao/RAFT/TUM_dataset/reading_t2/rgb/' + str(c) + '.png', frame)  #保存图片到指定目录，编号
    else:
        break
    c = c + 1

vc.release()
