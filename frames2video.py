import cv2
import os


image_folder='to_yuankai/OF_tracker_frames'
image_list=os.listdir(image_folder)
print(image_list)
image_num=len(image_list)
img = cv2.imread(os.path.join(image_folder,image_list[0]))
imginfo = img.shape
size = (imginfo[1], imginfo[0])  # 与默认不同，opencv使用 height在前，width在后，所有需要自己重新排序
print(size)

# 创建写入对象，包括 新建视频名称，每秒钟多少帧图片(10张) ,size大小
# 一般人眼最低分辨率为19帧/秒
fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #opencv3.0
videoWrite = cv2.VideoWriter( 'to_yuankai/OF_insufficient_example.mp4', fourcc, 20, size )

for i in range(image_num):
    filename = 'to_yuankai/OF_tracker_frames/' + str(i) + '.png'
    img = cv2.imread(filename)  # 1 表示彩图，0表示灰度图

    # 直接写入图片对应的数据
    videoWrite.write(img)

videoWrite.release()  # 关闭写入对象
print('end')
