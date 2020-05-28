import os
import cv2
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging
import json


def generate_categories(path):
    with open(path, 'r', encoding='utf-8') as f:
        names = f.read().splitlines()

    categories = []
    cat2id = {}
    for idx, name in enumerate(names):
        cat = {
            'id': idx + 1,
            'name': name
        }
        categories.append(cat)
        cat2id[name] = idx + 1

    return categories, cat2id


def generate_images(filename_txt_path, img_dir, target_img_dir):
    with open(filename_txt_path, 'r') as f:
        train_filenames = f.read().splitlines()

    images = []
    img2id = {}
    for idx, filename in tqdm(enumerate(train_filenames)):
        img_path = os.path.join(img_dir, filename + '.jpg')
        img_read = cv2.imread(img_path)
        shutil.copy(img_path, target_img_dir)

        img = {
            'id': idx,
            'file_name': filename + '.jpg',
            'height': img_read.shape[0],
            'width': img_read.shape[1]
        }

        images.append(img)
        img2id[filename] = idx

    return images, img2id


def generate_annotations(filename_txt_path, anno_dir, img2id, cat2id):
    annotations = []

    with open(filename_txt_path, 'r') as f:
        filenames = f.read().splitlines()

    anno_count = 0
    for filename in tqdm(filenames):
        anno_path = os.path.join(anno_dir, filename + '.xml')

        tree = ET.parse(anno_path)
        root = tree.getroot()
        objects = root.findall('object')

        for obj in objects:
            cat = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            anno = {
                'id': anno_count,
                'image_id': img2id[filename],
                'category_id': cat2id[cat],
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'iscrowd': 0
            }

            anno_count += 1
            annotations.append(anno)
    return annotations


def generate_coco_dataset(txt_dir, img_dir, anno_dir, coco_dir, flag, class_names_txt_path):
    """生成coco数据集

    Args:
        txt_dir: voc数据集中的train.txt和val.txt的存放路径
        img_dir: voc数据集中的图片存放路径
        anno_dir: voc数据集中的xml标注文件存放路径
        coco_dir: coco文件夹的路径
        flag: str. 'train' or 'val'
        class_names_txt_path: 存放所有类别名的文件路径

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
    )
    logging.info('handling {}...'.format(flag))

    # 1 路径组合
    txt_path = os.path.join(txt_dir, flag + '.txt')
    coco_img_dir = os.path.join(coco_dir, flag)
    coco_anno_dir = os.path.join(coco_dir, 'annotations')
    coco_anno_path = os.path.join(coco_anno_dir, 'instances_{}.json'.format(flag))

    # 2 若目标文件夹不存在，则创建
    if not os.path.exists(coco_img_dir):
        os.makedirs(coco_img_dir)

    if not os.path.exists(coco_anno_dir):
        os.makedirs(coco_anno_dir)

    # 3 生成json文件的三个部分
    # 3.1 生成categories，编号从1开始
    logging.info('generating categories...')
    categories, cat2id = generate_categories(class_names_txt_path)
    logging.info('categories generated\n')

    # 3.2 生成image
    logging.info('generating images...')
    images, img2id = generate_images(txt_path, img_dir, coco_img_dir)
    logging.info('images generated\n')

    # 3.3 生成annotations
    logging.info('generating annotations...')
    annotations = generate_annotations(txt_path, anno_dir, img2id, cat2id)
    logging.info('annotations generated\n')

    # 3.4 将三个组件组成dict写入json文件
    coco_dict = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(coco_anno_path, 'w', encoding='utf-8') as f:
        json.dump(coco_dict, f)

    logging.info('{} handled\n'.format(flag))
