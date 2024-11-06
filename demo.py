import os
import sys

# # 获取DINO文件夹的路径并添加到系统路径中
# dino_folder_path = "E:/Code/AlignCLIP/DINO"
# sys.path.insert(0, dino_folder_path)

import torch
import models_CLIP
from dataset.simple_tokenizer import tokenize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, Lambda
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    bicubic = InterpolationMode.BICUBIC
except ImportError:
    bicubic = Image.BICUBIC
from DINO.get_dino_features import get_dino_features
#* image preprocess
def _convert_to_rgb(image):
    return image.convert('RGB')
normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transforms = Compose([
    Resize(224, bicubic),
    CenterCrop(224),
    _convert_to_rgb,
    ToTensor(),
    normalize,
])

#* load model
model = models_CLIP.build_model('ViT-B-16') # RN50|ViT-B-32|ViT-B-16
ckpt_path = './pretrained_model/PyramidCLIP-143M-ViT-B16.pth' # specify path of checkpoint
if ckpt_path:
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])



#* input images
img = Image.open('E:\Code\AlignCLIP\image.jpg')
#* input texts
txt = ['hello world', 'good morning'] 


def get_img_features(model, img, normalize=True):
    img = transforms(img)
    if len(img.shape)==3: 
        img = img.unsqueeze(0)
    image_features = model.encode_image(img)
    if normalize:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


def get_txt_features(model, txt, normalize=True):
    txt = tokenize(txt)
    text_features = model.encode_text(txt)
    if normalize:
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_img_txt_logits(model, img, txt):
    image_features = get_img_features(model, img)
    text_features = get_txt_features(model, txt)
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    return logits_per_image, logits_per_text


# DINO_features = get_dino_features('E:\Code\AlignCLIP\image.jpg')
# print(DINO_features)
print(get_img_features(model, img))
print(get_txt_features(model, txt))
print(get_img_txt_logits(model, img, txt))
