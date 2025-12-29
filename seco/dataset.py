from PIL import ImageFilter
import random
import os
import json
import numpy as np
import torch
import cv2
import math
import xmltodict
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image, normalize, erase

from PIL import Image
from collections import OrderedDict, Counter
from multiprocessing import Manager


class COCODataset(Dataset):
    """
    Dataset to load data in COCO-style format and provide samples corresponding to individual objects.
    A sample consists of a target image (cropped to the objects bounding box), a context image (entire image),
    the bounding box coordinates of the target object ([xmin, ymin, w, h]) relative to the image size (e.g., (0.5,0.5)
    are the coords of the point in the middle of the image) and a label in [0,num_classes].
    """

    def __init__(self, annotations_file, image_dir, image_size, method='seco', idx2label=None, transform=None, target_transform=None, filtered_categories=None):
        """
        Args:
            annotations_file: path to COCO-style annotation file (.json)
            image_dir: path to the image folder
            image_size: desired size of the sample images, either a tuple (w,h) or an int if w=h
            idx2label: If a particular mapping between index and label is desired. Format: {idx: "labelname"}.
        """
        
        self.image_dir = image_dir
        self.image_size = image_size if type(image_size) == tuple else (image_size, image_size)
        self.transform = transform
        self.target_transform = target_transform
        self.filtered_categories = filtered_categories
        self.method = method

        with open(annotations_file) as f:
            self.coco_dict = json.load(f, object_pairs_hook=OrderedDict)

        
        self.annotations = self.coco_dict["annotations"]

        # filter according to categories
        if filtered_categories is not None:
            self.annotations = [a for a in self.annotations\
            if a["category_id"] in filtered_categories]
            self.coco_dict["categories"] = [c for c in self.coco_dict["categories"] if c["id"] in filtered_categories]
        print(self.coco_dict["categories"])
        self.imgid2id = {}
        self.id2file = {}
        for i, img in enumerate(self.coco_dict["images"]):
            self.id2file[img["id"]] = os.path.join(image_dir, img["file_name"])
            self.imgid2id[img["id"]] = i
        
        # filter out large-size object
        self.drop_unexisted()

        self.id2label = {} # maps label id to label name
        self.id2idx = {} # maps label id to index in 1-hot encoding
        self.label2id = {} # maps label name to label id
        if idx2label is None:
            self.idx2label = {} # maps index in 1-hot encoding to label name
            for idx, i in enumerate(self.coco_dict["categories"]):
                self.id2label[i["id"]] = i["name"]    
                self.idx2label[idx] = i["name"]
                self.id2idx[i["id"]] = idx
                self.label2id[i["name"]] = i["id"]
        else:
            assert(len(self.coco_dict["categories"]) == len(idx2label)), "Number of categorires in the annotation file does not agree with the number of categories in the custom idx2label mapping."
            
            self.idx2label = idx2label # maps index in 1-hot encoding to label name
            label2idx = {label: idx for idx, label in self.idx2label.items()}
            for i in self.coco_dict["categories"]:
                self.id2label[i["id"]] = i["name"]
                self.id2idx[i["id"]] = label2idx[i["name"]]
                self.label2id[i["name"]] = i["id"]
        
        self.NUM_CLASSES = len(self.id2label)


        # count annotations per class
        self.annotation_counts = Counter([a["category_id"] for a in self.annotations])
        self.annotation_counts = {self.id2idx[k]: v for k, v in self.annotation_counts.items()}
        self.named_annotation_counts = {self.idx2label[k]: v for k, v in self.annotation_counts.items()}
        self.relative_annotation_counts = np.array([self.annotation_counts[k] for k in sorted(self.annotation_counts.keys())])
        self.relative_annotation_counts = self.relative_annotation_counts / np.sum(self.relative_annotation_counts)
        self.relative_annotation_counts = torch.tensor(self.relative_annotation_counts, dtype=torch.float) # convert to tensor to simplify usage for reweighting
        
        print("-------------------------------\nAnnotation Counts\n-------------------------------")
        for k, v in self.named_annotation_counts.items():
            print("{0:10} {1:20} {2:10}".format(self.label2id[k], k, v))
        print("{0:20} {1:10}".format("Total", len(self.annotations)))
        print("-------------------------------\n")


    def __len__(self):
        return len(self.annotations)
    
    def drop_unexisted(self):
        new_anno = []
        print('''Following images don't exist:''')
        for anno in self.annotations:
            if os.path.exists(self.id2file[anno["image_id"]]):
                new_anno.append(anno)
            else:
                print(self.id2file[anno["image_id"]])
            
        self.annotations = new_anno

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # load label
        label = self.id2idx[annotation["category_id"]]

        # load image
        image = Image.open(self.id2file[annotation["image_id"]])
        image = image.convert("RGB")
        
        # compute bounding box coordinates relative to the image size
        xmin, ymin, w, h = annotation["bbox"]
        xmin_r, ymin_r, w_r, h_r = xmin / image.width, ymin / image.height, w / image.width, h / image.height
        img_w, img_h = image.width, image.height
        bbox_relative = torch.tensor([xmin_r, ymin_r, w_r, h_r])

        # crop to bounding box for target image
        target_image = image.crop((int(xmin), int(ymin), int(xmin + w), int(ymin + h)))
        
        image_t = to_tensor(image)

        bbox_int = list(map(lambda x: int(x), annotation["bbox"]))
        bbox_ccrop = [xmin_r,ymin_r,xmin_r+w_r,ymin_r+h_r]

        v = torch.zeros((3, bbox_int[3], bbox_int[2]))
        image_erased_t = erase(image_t, bbox_int[1], bbox_int[0], bbox_int[3], bbox_int[2], v)
        image_erased = to_pil_image(image_erased_t)

        target_image = target_image.resize(self.image_size)
        inputs = [image_erased, bbox_ccrop] if 'val' not in self.method else image_erased
        images_1 = self.transform(inputs) if self.transform is not None else image_erased 
        images_2 = self.target_transform(target_image) if self.target_transform is not None else target_image # objects
                          
        return images_1, images_2, bbox_relative, label

    
    def get_label_by_index(self, index):
        annotation = self.annotations[index]
        label = self.id2idx[annotation["category_id"]]
        return label


class COCODatasetImageBased(Dataset):
    """
    Dataset to load data in COCO-style format and provide samples corresponding to individual objects.
    A sample consists of a target image (cropped to the objects bounding box), a context image (entire image),
    the bounding box coordinates of the target object ([xmin, ymin, w, h]) relative to the image size (e.g., (0.5,0.5)
    are the coords of the point in the middle of the image) and a label in [0,num_classes].
    """

    def __init__(self, annotations_file, image_dir, image_size, bboxes=None, idx2label=None, transform=None, target_transform=None, method='selective_search',filtered_categories=None):
        """
        Args:
            annotations_file: path to COCO-style annotation file (.json)
            image_dir: path to the image folder
            image_size: desired size of the sample images, either a tuple (w,h) or an int if w=h
            idx2label: If a particular mapping between index and label is desired. Format: {idx: "labelname"}.
        """
        
        self.image_dir = image_dir
        self.image_size = image_size if type(image_size) == tuple else (image_size, image_size)
        self.transform = transform
        self.target_transform = target_transform
        self.method = method
        self.filtered_categories = filtered_categories

        with open(annotations_file) as f:
            self.coco_dict = json.load(f, object_pairs_hook=OrderedDict)

        self.annotations = self.coco_dict["annotations"]
        self.imgid2id = {}
        self.id2file = {}
        for i, img in enumerate(self.coco_dict["images"]):
            self.id2file[img["id"]] = os.path.join(image_dir, img["file_name"])
            self.imgid2id[img["id"]] = i
        # manager = Manager()
        if filtered_categories is not None:
            self.imgs = list(set([a["image_id"] for a in self.annotations\
            if a["category_id"] in filtered_categories]))
        else:
            self.imgs = list(set([a["image_id"] for a in self.annotations]))
        self.img_paths = np.array([self.id2file[i] for i in self.imgs])

        print('total imgs', len(self.imgs))
        self.bboxes = bboxes

    def __len__(self):
        return len(self.imgs)

    def randomCrop(self, height, width, scale=(0.2,0.5),ratio=(0.75, 1.3333333333333333)):
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return j, i, w, h
        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return j, i, w, h

    def __getitem__(self, idx):

        # imgid = self.id2file[self.imgs[idx]]
        imgid = self.img_paths[idx]
        # load image
        image = Image.open(imgid)
        image = image.convert("RGB")
        img_w, img_h = image.width, image.height

        if self.method == 'baseline':
            image_1 = self.transform(image)
            image_2 = self.target_transform(image) if self.target_transform is not None else self.transform(image)
            return image_1, image_2          
        elif self.method in ['dino_baseline','simclr_baseline']:
            return self.transform(image)

        if self.method == 'selective_search':
            if idx not in self.bboxes:
                bboxes = self.selective_search(image, img_w, img_h, img_w, img_h, 0, 0)
                self.bboxes[idx] = bboxes
            else:
                bboxes = self.bboxes[idx]

            bboxes_index_sampled = np.random.choice(len(bboxes), 4, replace=True)
            bboxes_sampled = [bboxes[i] for i in bboxes_index_sampled]
        elif self.method == 'random':
            bboxes_sampled = [self.randomCrop(img_h, img_w, (0.001,0.2), (0.2,5)) for _ in range(4)]


        erased_contexts, target_objects = [], []
        for xmin, ymin, w, h in bboxes_sampled:

            xmin_r, ymin_r, w_r, h_r = xmin / img_w, ymin / img_h, w / img_w, h / img_h
            
            bbox_relative = torch.tensor([xmin_r, ymin_r, w_r, h_r])

            # crop to bounding box for target image
            target_image = image.crop((int(xmin), int(ymin), int(xmin + w), int(ymin + h)))

            image_t = to_tensor(image)

            bbox_int = list(map(lambda x: int(x), [xmin, ymin, w, h]))
            bbox_ccrop = [xmin_r,ymin_r,xmin_r+w_r,ymin_r+h_r]

            v = torch.zeros((3, bbox_int[3], bbox_int[2]))
            image_erased_t = erase(image_t, bbox_int[1], bbox_int[0], bbox_int[3], bbox_int[2], v)
            image_erased = to_pil_image(image_erased_t)

            target_image = target_image.resize((96,96))
            inputs = [image_erased, bbox_ccrop] if 'val' not in self.method else image_erased
            context_erased = self.transform(inputs) if self.transform is not None else image_erased 
            target_object = self.target_transform(target_image) if self.target_transform is not None else target_image # objects
            erased_contexts.append(context_erased)
            target_objects.append(target_object)

        return torch.stack(erased_contexts), torch.stack(target_objects)
        

    def get_max_iou(self, pred_boxes, gt_box):
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

    def selective_search(self, img, w, h, o_w, o_h, i, j, res_size=224, reference=None):
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
        boxes = boxes[np.where(area_ratio < 0.1)[0]]
        width_ratio =  boxes[..., 2] / o_w
        boxes = boxes[np.where(width_ratio < 0.5)[0]]
        height_ratio = boxes[..., 3] / o_h
        boxes = boxes[np.where(height_ratio < 0.5)[0]]
        aspect_ratio = boxes[..., 2] / boxes[..., 3]
        boxes = boxes[np.where(aspect_ratio < 5)[0]]
        aspect_ratio = boxes[..., 2] / boxes[..., 3]
        boxes = boxes[np.where(aspect_ratio > 0.2)[0]]               
        
        proposals = [boxes[0]]
        for box in boxes:
            mIoU = self.get_max_iou(np.array(proposals), box)
    #         print(mIoU)
            if mIoU < 0.3:
                proposals.append(box)
        proposals = np.array(proposals) 
        return proposals.astype('int32')



class VOCDataset(Dataset):
    def __init__(self, anno_dir, img_dir, image_size=(224,224), idx2label=None, transform=None, target_transform=None, test_split_dir=None,method=None):
        self.anno_dir = anno_dir
        self.img_dir = img_dir
        self.idx2label = idx2label
        self.transform = transform
        self.image_size = image_size
        self.target_transform = target_transform
        self.test_split_dir = test_split_dir
        if idx2label is not None:
            self.label2idx = dict(zip(idx2label.values(),idx2label.keys()))
        self.make_annotations()
        
        print("-------------------------------\nAnnotation Counts\n-------------------------------")
        for k, v in self.named_annotation_counts.items():
            print("{0:10} {1:20} {2:10}".format(self.label2idx[k], k, v))
        print("{0:20} {1:10}".format("Total", len(self.annotations)))
        print("-------------------------------\n")

    def make_annotations(self):
        self.annotations = []
        self.named_annotation_counts = {}
        if self.test_split_dir is not None:
            with open(self.test_split_dir, 'r') as f:
                anno_list = [x.rstrip('\n')+'.xml' for x in f.readlines()]
        else:
            anno_list = os.listdir(self.anno_dir)
        print('Test Images:',len(anno_list))
        for anno_file in anno_list:
            full_path = os.path.join(self.anno_dir, anno_file)
            with open(full_path, 'r') as f:
                raw = ''.join(f.readlines())
                raw_anno = xmltodict.parse(raw)['annotation']
                filename = raw_anno['filename']
                image_size = raw_anno['size']
                for obj_dict in raw_anno['object']:
                    if not isinstance(obj_dict, dict):
                        continue 
                    xmin, ymin, xmax, ymax = [int(x) for x in obj_dict['bndbox'].values()]
                    bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
                    category_name = obj_dict['name']
                    category_idx = self.label2idx[category_name]
                    self.named_annotation_counts[category_name] = self.named_annotation_counts.get(category_name, 0) + 1
                    self.annotations.append({"image_name": filename, "id": category_idx, "bbox": bbox})
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # load label
        label = annotation["id"]
        filename = annotation["image_name"]
        # load image
        image = Image.open(os.path.join(self.img_dir,filename))
        image = image.convert("RGB")
        
        # compute bounding box coordinates relative to the image size
        xmin, ymin, w, h = annotation["bbox"]
        xmin_r, ymin_r, w_r, h_r = xmin / image.width, ymin / image.height, w / image.width, h / image.height
        img_w, img_h = image.width, image.height
        bbox_relative = torch.tensor([xmin_r, ymin_r, w_r, h_r])

        # crop to bounding box for target image
        target_image = image.crop((int(xmin), int(ymin), int(xmin + w), int(ymin + h)))

        image_t = to_tensor(image)
            # target_image = to_tensor(target_image)
        bbox_int = list(map(lambda x: int(x), annotation["bbox"]))
        bbox_ccrop = [xmin_r,ymin_r,xmin_r+w_r,ymin_r+h_r]
                # v = torch.empty((3, bbox_int[3], bbox_int[2]), dtype=torch.float32).normal_()

        v = torch.zeros((3, bbox_int[3], bbox_int[2]))
        image_erased_t = erase(image_t, bbox_int[1], bbox_int[0], bbox_int[3], bbox_int[2], v)
        image_erased = to_pil_image(image_erased_t)


        target_image = target_image.resize(self.image_size)
        images_1 = self.transform(image_erased) if self.transform is not None else image_erased 
        images_2 = self.target_transform(target_image) if self.target_transform is not None else target_image # objects
        return images_1, images_2, bbox_relative, label
    
    