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
from itertools import product
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from metric import AccuracyLogger
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image, normalize, erase
import torch.nn.functional as F
import cv2
import math 
import utils
import numpy as np
from collections import OrderedDict, Counter
from dataset import COCODataset
from PIL import Image
import matplotlib.pyplot as plt

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============

    if args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        # model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)


    model.cuda()
    state_dict = torch.load(args.pretrained_weights, map_location="cpu")['state_dict']
    state_dict = {k.replace("module.vam.backbone_context.", ""): v for k, v in state_dict.items() if 'backbone_context' in k}

    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))

    model.fc = nn.Identity()
    model.eval()
    # load weights to evaluate
    print(f"Model {args.arch} built.")


    print('embedding size:', embed_dim)
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    utils.load_pretrained_weights(linear_classifier, args.linear_weights, 'state_dict', args.arch)

    # ============ preparing data ... ============

    val_transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    label2idx  = {'wine glass':0,  'cup':1, 'knife':2, 'bowl':3, 'apple':4,
                  'cake':5, 'mouse':6, 'remote':7, 'keyboard':8, 'cell phone':9,
                 'microwave':10, 'book':11, 'toothbrush':12, 'pillow':13,'towel':14}

    object_prime(model, linear_classifier, label2idx, val_transform, args)

@torch.no_grad()
def object_prime(model, linear_classifier, label2idx, val_transform, args):
    patch_size = args.patch_size
    patch_num = (224 // args.patch_size) * 2 - 1
    with open(os.path.join(args.data_path,'metadata.json')) as f:
        metadata = json.load(f, object_pairs_hook=OrderedDict)
    priming_results = {}
    for image_id, data in metadata.items():
        image_path = os.path.join(args.data_path,image_id)
        absent_objs = data['absent_objs']
        image_name = image_id.split('.')[0]

        contexts, image_ori_t = make_contexts(image_path, val_transform, args)
        N = contexts.size(0)
        priming_probs = torch.zeros((len(absent_objs),224,224))
        logits = []
        i = -1
        for i in range(N // args.chunk):
            inputs = contexts[i*args.chunk: (i+1)*args.chunk].cuda()
            logits_tmp = linear_classifier(model(inputs))
            logits.append(logits_tmp)
        inputs = contexts[(i+1)*args.chunk:].cuda()
        logits_tmp = linear_classifier(model(inputs))
        logits.append(logits_tmp)
        logits = torch.concat(logits, dim=0)

        halfps = patch_size//2
        for k, label in enumerate(absent_objs): 
            idx = label2idx[label]
            attn_scores = F.softmax(logits[:,idx])#.permute(1,0)

            eval_train_map = attn_scores - attn_scores.min(0)[0]
            eval_train_map = eval_train_map / eval_train_map.max(0)[0]
            eval_train_map = eval_train_map.view((patch_num,patch_num)).cpu()
            eval_train_map = eval_train_map.view((patch_num,patch_num)).cpu()

            for j in range(patch_num):
                for i in range(patch_num):
                    i_start, i_end, j_start, j_end = int(i*0.5*patch_size), int((i*0.5+1)*patch_size), int(j*0.5*patch_size), int((j*0.5+1)*patch_size)
                    priming_probs[k, i_start:i_end, j_start:j_end] += (eval_train_map[j,i]) #/ (patch_size**2)
            priming_probs[k,halfps:224-halfps,0:halfps] /= 2
            priming_probs[k,0:halfps,halfps:224-halfps] /= 2
            priming_probs[k,halfps:224-halfps,224-halfps:224] /= 2
            priming_probs[k,224-halfps:224,halfps:224-halfps] /= 2
            priming_probs[k,halfps:224-halfps,halfps:224-halfps] /= 4

            visualization = image_ori_t * priming_probs[k]
            plt.axis('off')
            # plt.imsave(os.path.join(args.output_dir, '{}_{}_{}.png'.format(image_name,label,args.patch_size)),
            #     visualization.permute(1, 2, 0).detach().cpu().numpy())
            # plt.imsave(os.path.join(args.output_dir, '{}_{}_{}.pdf'.format(image_name,label,args.patch_size)),
            #     visualization.permute(1, 2, 0).detach().cpu().numpy(),dpi=96)          
        priming_results[image_id] = priming_probs
    torch.save(priming_results, os.path.join(args.output_dir,'{}_{}.pth'.format(args.method,args.patch_size)))
    



def make_contexts(img_path, transform, args):
    patch_num = (224 // args.patch_size) * 2 - 1
    images = []
    image = Image.open(img_path)
    image = image.convert("RGB")
    image = image.resize((224,224))
    image_t = to_tensor(image)
    for i, j in product(range(patch_num),range(patch_num)):     
        bbox_erase = np.array([i, j]) * args.patch_size // 2
        v = torch.zeros((3, args.patch_size, args.patch_size))
        # i -> w, j -> h 
        image_erased_t = erase(image_t, bbox_erase[1], bbox_erase[0], args.patch_size, args.patch_size, v)
        image_erased = to_pil_image(image_erased_t)
        image_erased = transform(image_erased)
        images.append(image_erased)
    return torch.stack(images), image_t


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
    parser.add_argument('--patch_size', default=28, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--linear_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--img_train_dir', type=str)
    parser.add_argument('--img_val_dir', type=str)
    parser.add_argument('--anno_dir', type=str)
    parser.add_argument('--method',default='seco', type=str)
    parser.add_argument('--gpu',default='0', type=str)
    parser.add_argument('--chunk', default=256, type=int)



    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    eval_linear(args)
