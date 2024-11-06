import argparse
import os
import math
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable
from utils import *
from datetime import datetime
import dataset
import models_CLIP
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, Lambda

from zero_shot_single import zero_shot_eval
device= "cuda"




def main():
    global args, input_resolution, test_datasets
    args.device = "cuda"
    args.workers = 0
    #* set random seed
    if args.seed > 0:
        set_seed(args.seed)
    else:
        cudnn.benchmark = True

    test_datasets = args.test_dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #* create model
    model = models_CLIP.build_model(args.visual_model).to(device)
    input_resolution = model.visual.input_resolution

    #* 设置交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    #* 评估模式下加载模型检查点
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            print('Invalid checkpoint: {}'.format(args.evaluate))
            return
        else:
            checkpoint = torch.load(args.evaluate, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint.get('epoch', -1)))

    if args.evaluate:
        test_dataset = list(args.test_dataset.split('+'))
        for idx in range(len(test_dataset)):
            args.test_dataset = test_dataset[idx]

            # from imagenetv2_pytorch import ImageNetV2Dataset
            # data_configs = {}
            # data_configs['CLIP'] = {
            #     'normalize': [(0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)],
            # }
            # normalize = Normalize(*data_configs['CLIP']['normalize'])
            # images = ImageNetV2Dataset(transform=normalize)
            # test_loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)

            data_loaders = dataset.load_data(args, input_resolution=input_resolution, _type='test')
            test_loader = data_loaders['test_loader'].dataloader
            with torch.no_grad():
                test_prec1, test_prec5 = test(test_loader, model, criterion, 0)
        return

def forward_test(data_loader, model, criterion, epoch, training=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.test_dataset == 'imagenet':
        zero_shot_metrics = zero_shot_eval(model, data_loader, epoch, args)
        top1, top5 = zero_shot_metrics['top1'], zero_shot_metrics['top5']
        print('ImageNet Zeroshot\t'
              'Prec@1/5 {top1:.2f}/{top5:.2f} \t'.format(top1=top1, top5=top5))
    elif args.test_dataset in ['dtd', 'flowers', 'cifar10', 'cifar100', 'car', 'pet', 'caltech', 'aircraft', 'food', 'sun', 'sat']:
        zero_shot_metrics = zero_shot_eval(model, data_loader, epoch, args)
        top1, top5 = zero_shot_metrics['top1'], zero_shot_metrics['top5']
        print('{} Zeroshot\t'
              'Prec@1/5 {top1:.2f}/{top5:.2f} \t'.format(args.test_dataset, top1=top1, top5=top5))
    return top1, top5

def test(data_loader, model, criterion, epoch):
    model.eval()
    return forward_test(data_loader, model, criterion, epoch, training=False)

if __name__ == '__main__':
    main()
