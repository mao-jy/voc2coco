import json
import logging

from utils import *


def main():
    """
    VOC数据集格式如下(后缀名应完全相同)：
        VOCdevkit/VOC/
            -ImageSets/
                -Main/
                    -train.txt
                    -val.txt
            -JPEGImages/
                -000000000001.jpg
                -000000000002.jpg
                -000000000003.jpg
            -Annotations
                -000000000001.xml
                -000000000002.xml
                -000000000003.xml

    生成的COCO数据集格式如下：
        -coco/
            -train/
                -000000000001.jpg
                -000000000002.jpg
                -000000000003.jpg
            -val/
                -000000000004.jpg
                -000000000005.jpg
                -000000000006.jpg
            -annotations
                -instances_train.json
                -instances_val.json

    """

    # TODO: 修改下面三个路径
    voc_dir = 'data/VOCdevkit/VOC'
    class_names_txt_path = 'data/train_classes.txt'
    coco_dir = 'data/coco'

    # 生成voc数据集相关路径
    txt_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
    img_dir = os.path.join(voc_dir, 'JPEGImages')
    anno_dir = os.path.join(voc_dir, 'Annotations')

    for flag in ['train', 'val']:
        generate_coco_dataset(txt_dir, img_dir, anno_dir, coco_dir, flag, class_names_txt_path)


if __name__ == "__main__":
    main()
