import torch
import math
from torch.utils.data import Dataset
from dataset import imagenet_classnames, openai_imagenet_template, tokenize, data_classnames, data_template
import os
import logging
from utils import *
import numpy as np
from DINO.get_dino_features import get_dino_features

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage
# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook the gradients and activations
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer = dict([*self.model.named_modules()])[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, class_idx):
        # Compute the gradients and activations
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Global average pooling
        cam = torch.sum(weights * self.activations, dim=1)  # Weighted sum of activations
        cam = torch.nn.functional.relu(cam)  # ReLU to remove negative values
        cam = cam[0].cpu().detach().numpy()  # Convert to numpy for visualization

        # Normalize the CAM to [0, 1]
        cam -= cam.min()
        cam /= cam.max()
        return cam

def apply_colormap_on_image(org_img, activation_map, colormap_name='jet'):
    """
    Apply a heatmap on the original image based on the activation map.
    """
    heatmap = plt.cm.get_cmap(colormap_name)(activation_map)
    heatmap = np.delete(heatmap, 3, 2)  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = ToPILImage()(heatmap)
    org_img = ToPILImage()(org_img)
    heatmap = heatmap.resize(org_img.size, resample=Image.BILINEAR)
    blended = Image.blend(org_img, heatmap, alpha=0.5)
    return blended

def visualize_grad_cam(model, dataloader, args, target_layer="layer4"):
    model.eval()
    grad_cam = GradCAM(model, target_layer)

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(args.device)
        labels = labels.to(args.device)

        # Forward pass
        outputs = model(images)
        predicted_class = torch.argmax(outputs, dim=1)

        # Backward pass for the predicted class
        outputs[:, predicted_class].backward()

        # Generate CAM
        cam = grad_cam.generate_cam(predicted_class)

        # Visualize heatmap
        for j in range(images.shape[0]):  # For batch
            activation_map = cam[j]
            blended_image = apply_colormap_on_image(images[j].cpu(), activation_map)
            plt.imshow(blended_image)
            plt.title(f"Class: {predicted_class[j]}, Label: {labels[j]}")
            plt.axis('off')
            plt.show()

        # Process only one batch for visualization
        break


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

    # Grad-CAM visualization
    visualize_grad_cam(model, dataloader, args, target_layer="layer4")  # Replace with your model's layer name

    return results
