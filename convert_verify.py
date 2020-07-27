import cv2
from pycocotools.coco import COCO
import os

json_path = 'data/coco/annotations/val.json'
img_dir = 'data/coco/val'
coco = COCO(json_path)

# 类别验证
categories = coco.cats
print('\n----------categories--------------')
for cat in categories.items():
    print(cat)

img_indices = coco.getImgIds()
for img_idx in img_indices:

    print('\n----------image - annotations--------------')
    print(coco.imgs[img_idx], '\n')

    img_name = coco.imgs[img_idx]['file_name']
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)

    assert (int(coco.imgs[img_idx]['height']) == img.shape[0])
    assert (int(coco.imgs[img_idx]['width']) == img.shape[1])

    for ann_idx in coco.getAnnIds(imgIds=img_idx):
        print(coco.anns[ann_idx])
        xmin = coco.anns[ann_idx]['bbox'][0]
        ymin = coco.anns[ann_idx]['bbox'][1]
        xmax = coco.anns[ann_idx]['bbox'][0] + coco.anns[ann_idx]['bbox'][2]
        ymax = coco.anns[ann_idx]['bbox'][1] + coco.anns[ann_idx]['bbox'][3]

        assert(int(coco.anns[ann_idx]['area']) == (xmax - xmin) * (ymax - ymin))

        cat_id = coco.anns[ann_idx]['category_id']
        cat_name = coco.cats[int(cat_id)]['name']

        cv2.namedWindow('test', 0)
        cv2.resizeWindow("test", 1960, 1080)

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        img = cv2.putText(img, cat_name, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)

    cv2.imshow('test', img)
    cv2.waitKey()
