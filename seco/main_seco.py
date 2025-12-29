#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pickle

from seco.builder import SeCoWithLoss, exclude_bias_and_norm
from pathlib import Path
from copy import deepcopy
from utils import GaussianBlur, Solarization, ContrastiveCrop, CCompose
from dataset import COCODatasetImageBased
from torch.utils.tensorboard import SummaryWriter

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='ocd', type=str)

parser.add_argument('--anno_dir', type=str)
parser.add_argument('--img_dir', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--base_lr', default=0.2, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr') # follow vicreg setting
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay') # follow vicreg setting
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--save_dir', default='checkpoints', type=str)
parser.add_argument('--save_every', default=10, type=int)
parser.add_argument('--sleep', default=0, type=float)

parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')
parser.add_argument("--K", default=100, type=int)
parser.add_argument("--memory_dim", default=128, type=int)
parser.add_argument("--memory_nhead", default=4, type=int)

def main():
    args = parser.parse_args()
    time.sleep(args.sleep)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    with open('bboxes/bboxes_{}.pkl'.format(args.dataset), 'rb') as f:
        bboxes = pickle.load(f)

    ngpus_per_node = torch.cuda.device_count()
    args.bboxes = bboxes

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = SeCoWithLoss(args)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256


    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            per_device_batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(model) # print model after SyncBatchNorm


    optim_params = [
        {'params': [p for p in model.module.parameters() if exclude_bias_and_norm(p) ], 'weight_decay': 0.0},
        {'params': [p for p in model.module.parameters() if not exclude_bias_and_norm(p) ], 'weight_decay': args.weight_decay}
    ]
    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum)
    fp16_scaler = torch.cuda.amp.GradScaler()
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            # args.start_epoch = checkpoint['epoch']
            # args.epochs += args.start_epoch 
            # msg = fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])
            # print(msg)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded fp16_scaler '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))
            # msg = model.load_state_dict(checkpoint['state_dict'],strict=False)
            # print(msg)
            # # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))

            adapted_ckpt = {}
            for key in checkpoint['model'].keys():
                if 'projector' not in key:
                    if 'backbone' in key: 
                        new_key = '.'.join([key.split('.')[0],'seco','backbone_context'] + key.split('.')[2:])
                        adapted_ckpt[new_key] = checkpoint['model'][key]
                        new_key = '.'.join([key.split('.')[0],'seco','backbone_object'] + key.split('.')[2:])
                        adapted_ckpt[new_key] = checkpoint['model'][key]
                    else:
                        new_key = '.'.join([key.split('.')[0],'seco'] + key.split('.')[1:])
                        adapted_ckpt[new_key] = checkpoint['model'][key]
                    
            msg = model.load_state_dict(adapted_ckpt,strict=False)
            print(msg)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = [
        ContrastiveCrop(2,5,size=224, scale=(0.4,1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_anno_path = os.path.join(args.anno_dir, args.dataset, 'train.json')
    transform = CCompose(augmentation)

    # no crop for object images
    target_augmentation = [
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ]
    target_transform = transforms.Compose(target_augmentation)

    if args.dataset == 'ocd':
        filtered_categories = None
    elif args.dataset == 'voc':
        filtered_categories=[1, 16, 17, 21, 18, 19, 20, 5, 2, 9, \
                        6, 3, 4, 7, 44, 62, 67, 64, 63, 72]

    train_dataset = COCODatasetImageBased(train_anno_path,
                                args.img_dir,
                                args.img_size,
                                transform=transform, 
                                target_transform=target_transform,
                                method='selective_search',
                                bboxes=args.bboxes,
                                filtered_categories=filtered_categories)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=per_device_batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    writer = SummaryWriter(args.save_dir+'_tb')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # train for one epoch
        train(train_loader, model, optimizer, epoch, args, writer, fp16_scaler)
        if (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'fp16_scaler': fp16_scaler.state_dict()
            }, is_best=False, filename='{}/checkpoint_lastest.pth.tar'.format(args.save_dir))

        if (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)) and (epoch+1) % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'fp16_scaler': fp16_scaler.state_dict()
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.save_dir,epoch))
    writer.close()

def train(train_loader, model, optimizer, epoch, args, writer, fp16_scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_component_names = ['total','repr','context_std','context_cov','object_std','object_cov']

    collapse_metrics_names = ['context_std', 'context_mr','context_mo','object_std', 
    'object_mr','object_mo','memory_std', 'memory_mr','memory_mo'] 
    loss_components = [AverageMeter('Loss/{}'.format(name), ':.4f') for name in loss_component_names]
    collapse_metrics = [AverageMeter('Collapse Metric/{}'.format(name), ':.4f') for name in collapse_metrics_names]
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time] + loss_components,
        prefix="Epoch: [{}]".format(epoch))
    prev_param = deepcopy(model.module.seco.ext_memory)
    # switch to train mode
    model.train()

    end = time.time()
    for i, (contexts_erased, objects) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)
        num_steps = epoch * len(train_loader) + i 
        lr = adjust_learning_rate(args, optimizer, len(train_loader), num_steps)

        if args.gpu is not None :
            objects = objects.cuda(args.gpu, non_blocking=True)
            contexts_erased = contexts_erased.cuda(args.gpu, non_blocking=True)

        contexts_erased = contexts_erased.permute(1,0,2,3,4)
        objects = objects.permute(1,0,2,3,4)

        with torch.cuda.amp.autocast(fp16_scaler is not None):

            loss, loss_scalars, x_metrics, y_metrics, m_metrics = model(contexts_erased.reshape((-1, 3, 224, 224)), objects.reshape((-1, 3, 96, 96)))

        for loss_component, loss_scalar in zip (loss_components, loss_scalars):
            loss_component.update(loss_scalar, contexts_erased.size(0))

        for collapse_metric, collapse_scalar in zip (collapse_metrics, x_metrics + y_metrics + m_metrics):
            collapse_metric.update(collapse_scalar, contexts_erased.size(0))
            
        # # compute gradient and do SGD step
        optimizer.zero_grad()

        fp16_scaler.scale(loss).backward()
        fp16_scaler.step(optimizer)
        fp16_scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if num_steps % args.print_freq == 0:
            for loss_component in loss_components:
                writer.add_scalar(loss_component.name, loss_component.avg, num_steps)
            for idx in [0,3,6]:
                writer.add_scalar(collapse_metrics[idx].name, collapse_metrics[idx].avg, num_steps)
            for idx in range(3):
                mr = collapse_metrics[idx*3+1]
                mo = collapse_metrics[idx*3+2]
                tag = mr.name.split('_')[0]
                writer.add_scalars(tag, {
                    mr.name.split('_')[1]:mr.avg,
                    mo.name.split('_')[1]:mo.avg
                }, num_steps)
                

            writer.add_scalar('Learning Rate', lr, num_steps)

            with torch.no_grad():
                current_param = model.module.seco.ext_memory
                l2 = torch.sum((current_param - prev_param) ** 2)
                prev_param = deepcopy(current_param)
            writer.add_scalar('Object Memory Params L2', l2, num_steps)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, extra_entries=[]):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += extra_entries
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(args, optimizer, num_iters, step):
    max_steps = args.epochs * num_iters
    warmup_steps = 10 * num_iters
    base_lr = args.lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == '__main__':
    main()
