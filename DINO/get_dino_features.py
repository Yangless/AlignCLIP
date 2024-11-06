
import os
import sys

# 获取DINO文件夹的路径并添加到系统路径中
dino_folder_path = "E:/Code/AlignCLIP/DINO"  # 修改成DINO文件夹的新路径
sys.path.insert(0, dino_folder_path)


import os
import torch
import numpy as np
from PIL import Image
import datasets.transforms as T
from main_demo import build_model_main
from util.slconfig import SLConfig


def load_dino_model():
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)

    # Define model config and checkpoint paths relative to the script location
    model_config_path = os.path.join(current_dir, "config/DINO/DINO_4scale_demo.py")
    model_checkpoint_path = os.path.join(current_dir, "ckpts/checkpoint0033_4scale.pth")

    # Load model configuration and build the model
    args = SLConfig.fromfile(model_config_path)
    args.device = 'cuda'
    model, criterion, postprocessors = build_model_main(args)

    # Load model checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model.cuda()

def transform_image_DINO(image):
    # # Load and preprocess the image
    #
    # transformss = T.Compose([
    #     T.RandomResize([800], max_size=1333),
    #     T.ToTensor(),
    #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # image, _ = transformss(image, None)
    # return image
    # 如果传入的是一个已加载的PIL图像或者Tensor，直接进行预处理
    # 确保 image 是 Tensor 类型
    if not isinstance(image, torch.Tensor):
        # 如果 image 不是 Tensor，我们假定它是 PIL 格式并进行转换
        if isinstance(image, Image.Image):
            transformss = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image, _ = transformss(image, None)
        else:
            raise TypeError("Expected input image to be a PIL Image or a Tensor.")
    else:
        # 如果传入的 image 已经是 Tensor，则直接进行归一化
        transformss = T.Compose([
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transformss(image)

    return image


def transform_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(image, None)
    return image




def get_dino_features(image):
    # Load model
    model = load_dino_model()

    # Transform image
    # image = transform_image_DINO(image)

    # Run model to get features
    with torch.no_grad():
        dino_features = model(image)

    return dino_features

if __name__ == "__main__":
    image_path = "./figs/idea.jpg"  # 替换成你想要的图片路径
    dino_features = get_dino_features(image_path)
