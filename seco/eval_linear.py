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
from dataset import COCODataset, VOCDataset

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
    if args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)


    model.cuda()
    state_dict_raw = torch.load(args.pretrained_weights, map_location="cpu")['state_dict']
    state_dict = {k.replace("module.vam.backbone_context.", ""): v for k, v in state_dict_raw.items() if 'vam.backbone_context' in k}

    if len(state_dict) == 0:
        state_dict = {k.replace("module.vam.backbone.", ""): v for k, v in state_dict_raw.items() if 'vam.backbone' in k }
    if len(state_dict) == 0:
        state_dict = {k.replace("module.backbone_context.", ""): v for k, v in state_dict_raw.items() if 'backbone_context' in k}

    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
        
    model.fc = nn.Identity()
    model.eval()
    print(f"Model {args.arch} built.")

 
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============


    train_transform = utils.CCompose([
        utils.ContrastiveCrop(2,5,size=224, scale=(0.4,1.0)),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_transform = pth_transforms.Compose([
            pth_transforms.Resize((224,224), interpolation=3),
            # pth_transforms.CenterCrop(22),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    target_transform = val_transform


    train_anno_path = os.path.join(args.anno_dir, args.dataset, 'train.json')
    val_anno_path = os.path.join(args.anno_dir, args.dataset, 'val.json')
    
    if args.dataset == 'ocd':
        filtered_categories = None


    elif args.dataset == 'voc':
        filtered_categories=[1, 16, 17, 21, 18, 19, 20, 5, 2, 9, \
                        6, 3, 4, 7, 44, 62, 67, 64, 63, 72]
    
    dataset_train = COCODataset(train_anno_path,
                                args.img_train_dir,
                                args.img_size,
                                transform=train_transform, 
                                target_transform=target_transform,
                                method='seco',
                                filtered_categories=filtered_categories)

    # idx2label: this is to make sure label indices between train set and test set are consistent
    dataset_val = COCODataset(val_anno_path,
                                args.img_val_dir,
                                args.img_size,
                                idx2label=dataset_train.idx2label, 
                                transform=val_transform, 
                                target_transform=target_transform,
                                method='seco_val',
                                filtered_categories=filtered_categories)


    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        state_dict = torch.load(args.linear_weights, map_location='cuda:{}'.format(args.gpu))['state_dict']
        msg = linear_classifier.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.linear_weights, msg))
        test_stats, acc_logger = validate_network(val_loader, model, linear_classifier, dataset_train.idx2label)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']*100:.2f}%")
        return

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    init_lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256. # linear scaling rule
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        init_lr, 
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )

    best_acc = 0
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats, acc_logger = validate_network(val_loader, model, linear_classifier, dataset_train.idx2label)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']*100:.2f}%")
            do_save = test_stats['acc1'] >= best_acc or epoch == args.epochs - 1
            save_suffix = 'best' if test_stats['acc1'] >= best_acc else 'last'
            best_acc = max(best_acc, test_stats['acc1'])
            print(f'Max accuracy so far: {best_acc*100:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if do_save: 
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc,
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_{}.pth.tar".format(save_suffix)))
                with open(os.path.join(args.output_dir, "acc_logger{}.pkl".format(save_suffix)),'wb') as f:
                    pickle.dump(acc_logger,f)
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.3f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (contexts, objects, _, label) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        contexts = contexts.cuda(non_blocking=True)
        objects = objects.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            feats = model(contexts)

        output = linear_classifier(feats)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, label)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
    parser = argparse.ArgumentParser('')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--dataset', default='ocd', type=str)
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--linear_weights', default='', type=str, help="Path to pretrained linear weights to evaluate.")
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
    parser.add_argument('--img_train_dir', type=str)
    parser.add_argument('--img_val_dir', type=str)
    parser.add_argument('--anno_dir', type=str)
    parser.add_argument('--method',default='seco', type=str)
    parser.add_argument('--gpu',default='0', type=str)


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    eval_linear(args)
