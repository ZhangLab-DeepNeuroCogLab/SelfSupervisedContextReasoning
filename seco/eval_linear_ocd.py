# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
import sys
import copy
import pickle
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from metric import AccuracyLogger

import math 
import utils
# from unsupervised.simsiam.simsiam.loader import COCODataset, COCODatasetWithIndex
from dataset import COCODataset
from PIL import Image


def eval_linear(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    if args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    state_dict = torch.load(args.pretrained_weights, map_location="cpu")['state_dict']
    state_dict = {k.replace("module.vam.backbone_context.", ""): v for k, v in state_dict.items() if 'backbone_context' in k}

    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    model.fc = nn.Identity()
    model.cuda()
    model.eval()
    # load weights to evaluate
    print(f"Model {args.arch} built.")


    print('embedding size:', embed_dim)
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    utils.load_pretrained_weights(linear_classifier, args.linear_weights, 'state_dict', args.arch)

    # ============ preparing data ... ============

    # if args.method == 'vicreg':

    val_transform = pth_transforms.Compose([
            pth_transforms.Resize((224,224), interpolation=3),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    target_transform = val_transform

    val_anno_path = os.path.join(args.anno_dir, 'val.json')

                                # erase=False)
    # idx2label: this is to make sure label indices between train set and test set are consistent
    dataset_val = COCODataset(val_anno_path,
                                args.img_val_dir,
                                args.img_size,
                                transform=val_transform, 
                                target_transform=target_transform,
                                method='vicreg_val')


    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)

    print(f"Data loaded with {len(dataset_val)} val imgs.")


    test_stats, acc_logger = validate_network(val_loader, model, linear_classifier, dataset_val.idx2label)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']*100:.2f}%")
    for k, v in test_stats.items():
        print('test',k,v)

@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, idx2label_dict=None):
    linear_classifier.eval()
    acc_logger = AccuracyLogger(idx2label_dict)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for contexts, objects, bbox_relative, label in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        contexts = contexts.cuda(non_blocking=True)
        objects = objects.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            feats = model(contexts)    
        output = linear_classifier(feats)
        loss = nn.CrossEntropyLoss()(output, label)

        bbox_areas = [w*h for _,_,w,h in bbox_relative]
        _, predictions = output.max(1, True)
        acc_logger.update(predictions, label, bbox_areas)

        batch_size = contexts.shape[0]
        metric_logger.update(loss=loss.item())


    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_stats['acc1'] = acc_logger.accuracy()
    test_stats['class_wise_acc'] = acc_logger.named_class_accuarcies()
    test_stats.update(acc_logger.bbox_items())
    return test_stats, acc_logger


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='resnet50', type=str, help='Architecture')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--linear_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--img_val_dir', type=str)
    parser.add_argument('--anno_dir', type=str)
    parser.add_argument('--gpu',default='0', type=str)


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    eval_linear(args)
