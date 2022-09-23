import cv2

# load video from your path
vc = cv2.VideoCapture('s1_t5_t4_track.mp4')
c = 0
rval = vc.isOpened()

while rval:
    c = c + 1
    rval, frame = vc.read()
    if rval:
        cv2.imwrite('to_yuankai/flow_result/' + str(c) + '.jpg', frame)
    else:
        break

vc.release()