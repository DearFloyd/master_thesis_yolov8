import cv2
import numpy as np


def draw_points(img_path):
    image = cv2.imread(img_path)

    coors = [[950, 75],  # 右上1
             [200, 200],  # 左上1
             [1250, 200],  # 右上2
             [120, 400],  # 左上2
             [1650, 500],  # 右上3
             [150, 900],  # 左上3
             [685, 125],  # 中右1
             [445, 160],  # 中左1
             [890, 260],  # 中右2
             [530, 330],  # 中左2
             [1365, 580],  # 中右3
             [745, 745],  # 中左3
             ]
    test_coors = [[ 950,   75],
                [1250,  200],
                [ 890,  260],
                [ 685,  125]]
    center = [[490, 450]]
    point_size = 10  
    point_color = (0, 0, 255) # BGR
    thickness = 8
    for coor in center:
        cv2.circle(image, (int(coor[0]),int(coor[1])), point_size, point_color, thickness)
    
    cv2.line(image, coors[0], coors[1], (0, 0, 255), 10)
    cv2.line(image, coors[2], coors[3], (0, 0, 255), 10)
    cv2.line(image, coors[4], coors[5], (0, 0, 255), 10)
    cv2.line(image, coors[6], coors[8], (0, 0, 255), 10)
    cv2.line(image, coors[7], coors[9], (0, 0, 255), 10)
    cv2.line(image, coors[8], coors[10], (0, 0, 255), 10)
    cv2.line(image, coors[9], coors[11], (0, 0, 255), 10)

    cv2.imwrite('/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/src_img/draw_2.png', image)


if __name__ == "__main__":
    img_path = '/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/src_img/2.png'
    draw_points(img_path)
    pass