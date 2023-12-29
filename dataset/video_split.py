import os
import cv2


def ExtractVideoFrame(video_input,output_path):
    """
    功能:将视频转成图片(提取视频的每一帧图片)
        1.能够设置多少帧提取一帧图片
        2.可以设置输出图片的大小及灰度图
        3.手动设置输出图片的命名格式
    """
    # 输出文件夹不存在，则创建输出文件夹
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    times = 29000               # 用来记录帧
    frame_frequency = 50    # 提取视频的频率，每frameFrequency帧提取一张图片，提取完整视频帧设置为1
    count = 0               # 计数用，分割的图片按照count来命名
    cap = cv2.VideoCapture(video_input)  # 读取视频文件
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(times))
    print('开始提取', video_input, '视频的图片')
    while True:
        times += 1
        res, image = cap.read()          # 读出图片。res表示是否读取到图片，image表示读取到的每一帧图片
        if not res:
            print('图片提取结束')
            break
        if times % frame_frequency == 0:
            # picture_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图片转成灰度图
            # image_resize = cv2.resize(image, (368, 640))            # 修改图片的大小
            img_name = '1920x1080_1016_' + str(count).zfill(6)+'.jpg'
            cv2.imwrite(output_path + os.sep + img_name, image)
            count += 1

            print(output_path + os.sep + img_name)  # 输出提示
    cap.release()


def ShowSpecialFrame(file_path,frame_index):
    """
    功能:获取视频的指定帧并进行显示
    """
    cap = cv2.VideoCapture(file_path)  # 读取视频文件
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    if cap.isOpened():  # 判断是否正常打开
        rval, frame = cap.read()
        cv2.imshow("image:"+frame_index,frame)
        cv2.waitKey()
    cap.release()


def ExtractVideoBySpecialFrame(video_input,output_path,start_frame_index,end_frame_index = -1):
    """
    功能:切割视频的指定帧。比如切割视频从100帧到第200帧的图片
        1.能够设置多少帧提取一帧图片
        2.可以设置输出图片的大小及灰度图
        3.手动设置输出图片的命名格式
    """
    # 输出文件夹不存在，则创建输出文件夹
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    cap = cv2.VideoCapture(video_input) # 读取视频文件
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame_index))    # 从指定帧开始读取文件
    times = 0                           # 用来记录帧
    frame_frequency = 10                # 提取视频的频率，每frameFrequency帧提取一张图片，提取完整视频帧设置为1
    count = 0                           # 计数用，分割的图片按照count来命名

    # 未给定结束帧就从start_frame_index帧切割到最后一帧
    if end_frame_index == -1:
        print('开始提取', video_input, '视频从第',start_frame_index,'帧到最后一帧的图片！！')
        while True:
            times += 1
            res, image = cap.read()  # 读出图片
            if not res:
                print('图片提取结束！！')
                break
            if times % frame_frequency == 0:
                # picture_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图片转成灰度图
                # image_resize = cv2.resize(image, (368, 640))            # 修改图片的大小
                img_name = str(count).zfill(6) + '.jpg'
                cv2.imwrite(output_path + os.sep + img_name, image)
                count += 1
                print(output_path + os.sep + img_name)  # 输出提示
    else:
        print('开始提取', video_input, '视频从第', start_frame_index, '帧到第',end_frame_index,'帧的图片！！')
        k = end_frame_index - start_frame_index + 1
        while(k >= 0):
            times += 1
            k -= 1
            res, image = cap.read()  # 读出图片
            if not res:
                print('图片提取结束！！')
                break
            if times % frame_frequency == 0:
                # picture_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图片转成灰度图
                # image_resize = cv2.resize(image, (368, 640))            # 修改图片的大小
                img_name = str(count).zfill(6) + '.jpg'
                cv2.imwrite(output_path + os.sep + img_name, image)
                count += 1
                print(output_path + os.sep + img_name)  # 输出提示
        print('图片提取结束！！')
    cap.release()


def split_video_period(video_path):
    pass


if __name__ == "__main__":
    # 视频路径
    video_input = 'G:/classroom_dataset/10.22/001.mp4'
    # 图片输出路径
    output_path = 'G:/classroom_dataset/image_10_16'

    # 提取视频图片
    ExtractVideoFrame(video_input, output_path)

    # 显示视频第100帧的图片
    # ShowSpecialFrame(video_input, 1500)

    # 获取视频第100帧到第200帧的图片
    # ExtractVideoBySpecialFrame(video_input, output_path, 100, 200)
