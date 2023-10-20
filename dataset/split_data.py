import os, shutil, random
random.seed(0)
import numpy as np
from sklearn.model_selection import train_test_split

val_size = 0.1
test_size = 0.2
postfix = 'jpg'
imgpath = '/workspace/cv-docker/joey04.li/datasets/datasets_1500+100/img_10.20_1500+100_current_final'
txtpath = '/workspace/cv-docker/joey04.li/datasets/datasets_1500+100/annotation_10.20_yolo_format_current_final'

os.makedirs('/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/images/train', exist_ok=True)
os.makedirs('/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/images/val', exist_ok=True)
os.makedirs('/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/images/test', exist_ok=True)
os.makedirs('/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/labels/train', exist_ok=True)
os.makedirs('/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/labels/val', exist_ok=True)
os.makedirs('/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/labels/test', exist_ok=True)

listdir = np.array([i for i in os.listdir(txtpath) if 'txt' in i])
random.shuffle(listdir)
train, val, test = listdir[:int(len(listdir) * (1 - val_size - test_size))], listdir[int(len(listdir) * (1 - val_size - test_size)):int(len(listdir) * (1 - test_size))], listdir[int(len(listdir) * (1 - test_size)):]
print(f'train set size:{len(train)} val set size:{len(val)} test set size:{len(test)}')

for i in train:
    if i == 'classes.txt':
        continue
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/images/train/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/labels/train/{}'.format(i))

for i in val:
    if i == 'classes.txt':
        continue
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/images/val/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/labels/val/{}'.format(i))

for i in test:
    if i == 'classes.txt':
        continue
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/images/test/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '/workspace/cv-docker/joey04.li/datasets/classroom_datasets_1020/labels/test/{}'.format(i))