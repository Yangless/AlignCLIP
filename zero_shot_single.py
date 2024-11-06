import torch
import math
from torch.utils.data import Dataset
from dataset import imagenet_classnames, openai_imagenet_template, tokenize, data_classnames, data_template
import os
import logging
from utils import *
import numpy as np
from DINO.get_dino_features import get_dino_features
def get_text_features(model, classnames, templates, args):
    N = len(classnames)
    with torch.no_grad():
        text_features_split = []
        for classname in classnames:
            texts = [template(classname) for template in templates]
            texts = tokenize(texts).to(args.device)
            class_embeddings = model.encode_text(texts)
            logits = model.logit_scale
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
            text_features_split.append(class_embedding)
        text_features = torch.stack(text_features_split, dim=0)
        return text_features.to(args.device)

def get_text_features(model, classnames, templates, args):
    N = len(classnames)
    with torch.no_grad():
        text_features_split = []
        for classname in classnames:
            texts = [template(classname) for template in templates]
            texts = tokenize(texts).to(args.device)
            class_embeddings = model.encode_text(texts)
            logits = model.logit_scale
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
            text_features_split.append(class_embedding)
        text_features = torch.stack(text_features_split, dim=0)
        return text_features.to(args.device)


def get_image_features(model, dataloader, args):
    image_features = []
    for i, (image, target) in enumerate(dataloader):
        image = image.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        # SKT
        # from PIL import Image
        # import DINO.datasets.transforms as T1
        # image1 = Image.open("E:/Code/DINO-main/figs/idea.jpg").convert("RGB")  # load image
        #
        # # transform images
        # transforms = T1.Compose([
        #     T1.RandomResize([800], max_size=1333),
        #     T1.ToTensor(),
        #     T1.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # image2, _ = transforms(image1, None)


        DINO_embeddings = get_dino_features(image.cuda())
        # print("DINO_embeddings.shape",DINO_embeddings.shape)

        image_embeddings = model.encode_image(image)
        # print("image_embeddings.shape", image_embeddings.shape)
        #* logits are already multiplied on text features
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        # print("image_embeddings.shape",image_embeddings.shape)
        Align_embeddings = model.Align_encoder(image_embeddings,DINO_embeddings)
        # print(Align_embeddings)

        image_features.append(torch.cat([image_embeddings, target.unsqueeze(1)], dim=1))
    image_features = torch.cat(image_features, dim=0)
    return image_features.to(args.device)

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True)) for k in topk]

def run(model, text_features, image_features, dataloader, args):
    with torch.no_grad():
        logits = image_features[:, :-1] @ text_features.to(image_features.device).t()
        target = image_features[:, -1]

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        acc1 = torch.tensor(acc1).to(args.device)
        acc5 = torch.tensor(acc5).to(args.device)

    n = len(image_features)
    acc1 = (acc1 / n) * 100.
    acc5 = (acc5 / n) * 100.
    return acc1, acc5

class loader(Dataset):
    def __init__(self, path):
        data = torch.load(path)

def zero_shot_eval(model, dataloader, epoch, args):
    if args.test_dataset == 'imagenet':
        text_features = get_text_features(model, imagenet_classnames, openai_imagenet_template, args)
    else:
        text_features = get_text_features(model, data_classnames[args.test_dataset], openai_imagenet_template, args)
    image_features = get_image_features(model, dataloader, args)

    results = {}
    top1, top5 = run(model, text_features, image_features, dataloader, args)
    results['top1'] = top1
    results['top5'] = top5
    return results
