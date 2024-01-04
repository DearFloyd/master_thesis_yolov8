import cv2
import numpy as np


def draw_points(img_path):
    image = cv2.imread(img_path)

    coors = [[900, 125],  # 右上1
             [200, 250],  # 左上1
             [1150, 200],  # 右上2
             [180, 400],  # 左上2
             [1450, 400],  # 右上3
             [150, 700]# 左上3
             ]
    
    point_size = 10  
    point_color = (0, 0, 255) # BGR
    thickness = 8
    for coor in coors:
        cv2.circle(image, (int(coor[0]),int(coor[1])), point_size, point_color, thickness)
    
    cv2.imwrite('/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/src_img/draw_2.png', image)


if __name__ == "__main__":
    img_path = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/src_img/2.png'
    draw_points(img_path)
    pass