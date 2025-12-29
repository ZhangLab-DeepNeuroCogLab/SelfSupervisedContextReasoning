import concurrent.futures
from functools import partial
import sys
from dataset import COCODatasetImageBased
from PIL import Image
import cv2
import numpy as np
import pickle
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--anno_dir', type=str)
parser.add_argument('--img_dir', type=str)
parser.add_argument('--dataset', default='ocd', type=str)
args = parser.parse_args()

def get_max_iou(pred_boxes, gt_box):
    """
        pred_boxes : multiple coordinate for predict bounding boxes (x, y, w, h)
        gt_box :   the coordinate for ground truth bounding box (x, y, w, h)
        return :   the max iou score about pred_boxes and gt_box
    """
    # 1.get the coordinate of inters
    ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
    ixmax = np.minimum(pred_boxes[:, 0] + pred_boxes[:, 2], gt_box[0] + gt_box[2])
    iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
    iymax = np.minimum(pred_boxes[:, 1] + pred_boxes[:, 3], gt_box[1] + gt_box[3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = (pred_boxes[:, 2] * pred_boxes[:, 3] + gt_box[2] * gt_box[3] - inters)

    # 4. calculate the overlaps and find the max overlap between pred_boxes and gt_box
    iou = inters / uni
    iou_max = np.max(iou)

    return iou_max

def selective_search(img, w, h, o_w, o_h, i, j, res_size=224, reference=None):
    img_det = np.array(img)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    if res_size is not None:
        img_det = cv2.resize(img_det, (res_size, res_size))

    ss.setBaseImage(img_det)
    ss.switchToSelectiveSearchFast()
    boxes = ss.process().astype('float32')

    if res_size is not None:
        boxes /= res_size
        boxes *= np.array([w, h, w, h])
        

    area_ratio = boxes[..., 2] * boxes[..., 3] / (o_h * o_w)
    boxes = boxes[np.where(area_ratio < 0.2)[0]]
    aspect_ratio = boxes[..., 2] / boxes[..., 3]
    boxes = boxes[np.where(aspect_ratio < 5)[0]]
    aspect_ratio = boxes[..., 2] / boxes[..., 3]
    boxes = boxes[np.where(aspect_ratio > 0.2)[0]]               
            
    proposals = [boxes[0]]
    for box in boxes:
        mIoU = get_max_iou(np.array(proposals), box)
        if mIoU < 0.3:
            proposals.append(box)
    proposals = np.array(proposals) 
    return proposals.astype('int32')

def process_one_image(idx, imgs, id2file):
    print('processing: ',idx)
    imgid = id2file[imgs[idx]]
    # load image
    image = Image.open(imgid)
    image = image.convert("RGB")
    img_w, img_h = image.width, image.height      
    bboxes = selective_search(image, img_w, img_h, img_w, img_h, 0, 0)
    return (idx, bboxes)

if args.dataset == 'ocd':
    filtered_categories = None
elif args.dataset == 'voc':
    filtered_categories=[1, 16, 17, 21, 18, 19, 20, 5, 2, 9, \
                    6, 3, 4, 7, 44, 62, 67, 64, 63, 72]

train_anno_path = os.path.join(args.anno_dir, 'train.json')
train_dataset = COCODatasetImageBased(train_anno_path,
                                args.img_dir,
                                224,
                                filtered_categories)
indices = list(range(len(train_dataset)))
chunksize = 125

func = partial(process_one_image, imgs=train_dataset.imgs, id2file=train_dataset.id2file)

with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
    bboxes = executor.map(func, indices, chunksize=chunksize)
bboxes = dict(list(bboxes))
with open('bboxes/bboxes_{}.pkl'.format(dataset), 'wb') as f:
    pickle.dump(bboxes, f)