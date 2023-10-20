import os
import shutil

# 根据label来匹配image 对齐数据与标签
label_dir = '/workspace/cv-docker/joey04.li/datasets/datasets_1500+100/annotation_10.20_yolo_format'
source_img_dir = '/workspace/cv-docker/joey04.li/datasets/datasets_1500+100/img_10.16_100'
output_img_dir = '/workspace/cv-docker/joey04.li/datasets/datasets_1500+100/img_10.20_1500+100'

label_list = os.listdir(label_dir)
for label_name in label_list:
    name, ext = os.path.splitext(label_name)
    if name == 'classes' and ext == '.txt':
        continue
    elif ext == '.txt':
        source_img_path = os.path.join(source_img_dir, name + '.jpg')
        if os.path.exists(source_img_path):
            shutil.copy2(source_img_path, output_img_dir)

    