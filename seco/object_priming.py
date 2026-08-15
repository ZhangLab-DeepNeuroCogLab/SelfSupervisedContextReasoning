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
import unsupervised.dino.vision_transformer as vits
import numpy as np
from collections import OrderedDict, Counter
# from unsupervised.simsiam.simsiam.loader import COCODataset, COCODatasetWithIndex
from dataset import COCODataset
from PIL import Image
import matplotlib.pyplot as plt
def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        # model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    if args.object_stream_model is not None:
        object_model = copy.deepcopy(model)
        if args.method == 'simsiam':
            object_model = utils.load_simsiam_ckpt(args, object_model, args.object_stream_model)
        elif args.method == 'vicreg':
            object_model = utils.load_vicreg_ckpt(args, object_model, args.object_stream_model)
        else:
            utils.load_pretrained_weights(object_model, args.object_stream_model, args.checkpoint_key, args.arch, args.patch_size)
        
        object_model.fc = nn.Identity()
        object_model.cuda()
        object_model.eval()
    else:
        object_model = None

    model.cuda()
    if args.method == 'simsiam':
        # model = utils.load_pretrained_context_ckpt(args, model)
        model = utils.load_simsiam_ckpt(args, model)
    elif args.method == 'vicreg':
        model = utils.load_vicreg_ckpt(args, model)
    elif args.method == 'ours':
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")['state_dict']
        state_dict = {k.replace("module.vam.backbone_context.", ""): v for k, v in state_dict.items() if 'backbone_context' in k}
        # remove `backbone.` prefix induced by multicrop wrapper
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.fc = nn.Identity()
    model.eval()
    # load weights to evaluate
    # utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")


    print('embedding size:', embed_dim)
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    utils.load_pretrained_weights(linear_classifier, args.linear_weights, 'state_dict', args.arch, args.patch_size)

    # ============ preparing data ... ============

    val_transform = pth_transforms.Compose([
            # pth_transforms.Resize((224,224), interpolation=3),
            # pth_transforms.CenterCrop(22),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    label2idx  = {'wine glass':0,  'cup':1, 'knife':2, 'bowl':3, 'apple':4,
                  'cake':5, 'mouse':6, 'remote':7, 'keyboard':8, 'cell phone':9,
                 'microwave':10, 'book':11, 'toothbrush':12, 'pillow':13,'towel':14}

    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
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

            # # new strategy
            # eval_train_map = logits[:,idx].view((patch_num,patch_num)).cpu()
            # print(eval_train_map.max())

            # print('eval_train_map',eval_train_map.permute(1,0),eval_train_map.shape)
            for j in range(patch_num):
                for i in range(patch_num):
                    i_start, i_end, j_start, j_end = int(i*0.5*patch_size), int((i*0.5+1)*patch_size), int(j*0.5*patch_size), int((j*0.5+1)*patch_size)
                    # print(i,j,i_start,':',i_end, j_start,':',j_end)
                    priming_probs[k, i_start:i_end, j_start:j_end] += (eval_train_map[j,i]) #/ (patch_size**2)
            priming_probs[k,halfps:224-halfps,0:halfps] /= 2
            priming_probs[k,0:halfps,halfps:224-halfps] /= 2
            priming_probs[k,halfps:224-halfps,224-halfps:224] /= 2
            priming_probs[k,224-halfps:224,halfps:224-halfps] /= 2
            priming_probs[k,halfps:224-halfps,halfps:224-halfps] /= 4
            # new strategy
            # priming_probs[k] = F.softmax(priming_probs[k].view((-1))).view((224,224))

            visualization = image_ori_t * priming_probs[k]
            plt.axis('off')
            # plt.imsave(os.path.join(args.output_dir, '{}_{}_{}.png'.format(image_name,label,args.patch_size)),
            #     visualization.permute(1, 2, 0).detach().cpu().numpy())
            # plt.imsave(os.path.join(args.output_dir, '{}_{}_{}.pdf'.format(image_name,label,args.patch_size)),
            #     visualization.permute(1, 2, 0).detach().cpu().numpy(),dpi=96)          
            # assert 1 == 2
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
        # print(i, j, bbox_erase)
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
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=28, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--linear_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.01, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
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
    parser.add_argument('--drop_threshold', default=0.1, type=float)
    parser.add_argument('--img_train_dir', type=str)
    parser.add_argument('--img_val_dir', type=str)
    parser.add_argument('--anno_dir', type=str)
    parser.add_argument('--concat', default=False, type=bool)
    parser.add_argument('--object_stream_model',default=None, type=str)
    parser.add_argument('--only_object',default=False,type=bool,help='only used in ablation study')
    parser.add_argument('--method',default='simsiam', type=str)
    parser.add_argument('--gpu',default='0', type=str)
    parser.add_argument('--chunk', default=256, type=int)



    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    eval_linear(args)
