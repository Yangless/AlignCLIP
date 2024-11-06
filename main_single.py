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


from zero_shot_single import *
device= "cuda"
criterion = nn.CrossEntropyLoss()

def train(train_loader, model, optimizer, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for i, (images, texts) in enumerate(train_loader):
        # len(texts)
        images = images.to(device)
        texts = texts.to(device)
        #
        print(f"Iteration {i + 1}")
        # # Encode image and text features
        # image_features = model.encode_image(images)
        # text_features = model.encode_text(texts)
        if args.test_dataset == 'imagenet':
            text_features = get_text_features(model, imagenet_classnames, openai_imagenet_template, args)
        else:
            text_features = get_text_features(model, data_classnames[args.test_dataset], openai_imagenet_template, args)
        image_features = model.encode_image(images)
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # # Compute logits
        # logits_per_image = image_features @ text_features.T
        # logits_per_text = text_features @ image_features.T
        #
        # # Compute loss
        # labels = torch.arange(len(images)).to(device)
        # loss_i = criterion(logits_per_image, labels)
        # loss_t = criterion(logits_per_text, labels)
        # loss = (loss_i + loss_t) / 2

        # Compute logits
        logits_per_image = image_features @ text_features.T

        # For text-to-image, select the top 16 rows with the highest average similarity scores
        logits_per_text = text_features @ image_features.T
        row_means = logits_per_text.mean(dim=1)  # Calculate mean similarity score for each row
        #fix bug ,when last batch size is less than 16
        top_k_indices = torch.topk(row_means, k=len(texts), dim=0).indices  # Get indices of top 16 rows
        top_k_logits_per_text = logits_per_text[top_k_indices, :]  # Select top 16 rows
        # print(logits_per_text.shape,top_k_logits_per_text.shape)
        # Compute loss
        labels = torch.arange(len(images)).to(device)
        loss_i = criterion(logits_per_image, labels)
        loss_t = criterion(top_k_logits_per_text, labels)  # Use the top 16 logits for text-to-image loss

        # Average the two losses
        loss = (loss_i + loss_t) / 2

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track accuracy and loss
        total_loss += loss.item()
        _, preds = logits_per_image.max(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(images)

    train_loss = total_loss / len(train_loader)
    train_accuracy = 100.0 * total_correct / total_samples
    print(f"Epoch [{epoch+1}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    return train_loss, train_accuracy


def main():
    global args, input_resolution, test_datasets
    args.device = "cuda"
    args.workers = 0

    # Set random seed for reproducibility
    if args.seed > 0:
        set_seed(args.seed)
    else:
        cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models_CLIP.build_model(args.visual_model).to(device)
    input_resolution = model.visual.input_resolution
    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    # scaler = GradScaler() if args.mixed_precision else None


    # Load training and validation datasets
    data_loaders = dataset.load_data(args, input_resolution=input_resolution, _type='test')
    train_loader = data_loaders['train_loader'].dataloader
    val_loader = data_loaders['val_loader'].dataloader

    # Training loop
    # for epoch in range(args.epochs):
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_accuracy = train(train_loader, model, optimizer, epoch)

        # Validate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_top1, val_top5 = forward_test(val_loader, model, criterion, epoch)
            print(f"Validation Prec@1: {val_top1:.2f}% Prec@5: {val_top5:.2f}%")

        # Save checkpoint after every epoch
        args.checkpoint_dir = "./ckpts"
        if args.checkpoint_dir:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")


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
