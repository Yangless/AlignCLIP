import torchvision.transforms as transforms
import numpy as np
from transformers import CLIPTokenizer
import torch.backends.cudnn as cudnn
from utils import *

import models_CLIP

from zero_shot_single import zero_shot_eval
device= "cuda"
# 使用 CLIPTokenizer 对标签进行编码
tokenizer =  CLIPTokenizer.from_pretrained("./clip-vit-base-patch32/")
#text_model = CLIPTextModel.from_pretrained("./clip-vit-base-patch32/")
eos_token_id = tokenizer.eos_token_id

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import xml.etree.ElementTree as ET

# 定义动作类别
ACTIONS = [
    "applauding",
    "blowing_bubbles",
    "brushing_teeth",
    "cleaning_the_floor",
    "climbing",
    "cooking",
    "cutting_trees",
    "cutting_vegetables",
    "drinking",
    "feeding_a_horse",
    "fishing",
    "fixing_a_bike",
    "fixing_a_car",
    "gardening",
    "holding_an_umbrella",
    "jumping",
    "looking_through_a_microscope",
    "looking_through_a_telescope",
    "playing_guitar",
    "playing_violin",
    "pouring_liquid",
    "pushing_a_cart",
    "reading",
    "phoning",
    "riding_a_bike",
    "riding_a_horse",
    "rowing_a_boat",
    "running",
    "shooting_an_arrow",
    "smoking",
    "taking_photos",
    "texting_message",
    "throwing_frisby",
    "using_a_computer",
    "walking_the_dog",
    "washing_dishes",
    "watching_TV",
    "waving_hands",
    "writing_on_a_board",
    "writing_on_a_book"
]


# 自定义数据集类
class Stanford40Dataset(Dataset):
    def __init__(self, root_dir, annotations_dir, transform=None):
        self.root_dir = root_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_paths, self.labels = self.load_annotations()

    def load_annotations(self):
        image_paths = []
        labels = []
        for filename in os.listdir(self.annotations_dir):
            if not filename.endswith('.xml'):
                continue
            xml_path = os.path.join(self.annotations_dir, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            image_file = root.find('filename').text
            action = root.find('.//action').text

            image_paths.append(os.path.join(self.root_dir, image_file))
            labels.append(ACTIONS.index(action))

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, ACTIONS[label],label

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((320, 320)),   #960, 640
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, model_output, text_embedding):
        return nn.functional.mse_loss(model_output, text_embedding)



# 定义训练函数
def test_model(model, dataloader, criterion, optimizer, device):
    model.to(device)
    with torch.no_grad():
       # model.train()
       # for images, texts, labels in dataloader:
        for images, texts , label in dataloader:



            images = images.to(device)
            #images  reshape
            #归一化  images = transform(image).unsqueeze(0)  # 假设 transform 已经定义
            # RGBA 转换  image = Image.fromarray(image).convert("RGBA")

            # texts = texts.to(device)
            # 每次输入几个相似texts，最大化labels
            texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")['input_ids']
            texts = texts.to(dtype=torch.int32)
            texts = texts.to(device)

            label = label.to(device)
            optimizer.zero_grad()

            # 前向传播
            output = model(images, texts)

            output = output.float()
            label = label.float()
           # print(output,label,"output,label")
            #print("images_path",img_path)
            print("output",output,"label", label)
            loss = criterion(output,label)
            print("loss",loss)



# 初始化模型、损失函数和优化器

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# * create model
model = models_CLIP.build_model(args.visual_model).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 创建数据集和数据加载器
root_dir = 'D:/Y/Stanford40/JPEGImages'
annotations_dir = 'D:/Y/Stanford40/XMLAnnotations'



dataset = Stanford40Dataset(root_dir=root_dir, annotations_dir=annotations_dir, transform=transform)

# 随机选择一部分数据进行加载
num_samples = 1000  # 要加载的数据样本数
indices = np.random.choice(len(dataset), num_samples, replace=False)
from torch.utils.data.sampler import SubsetRandomSampler
sampler = SubsetRandomSampler(indices)

dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
weights_path = "Lang—ARmodel_epoch_2_loss_1736.764264240265.pth"
#dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
model.load_state_dict(torch.load(weights_path))


print("len_dataset",len(dataset),"len_dataloader",len(dataloader))
# 示例：读取一个批次的数据
# for images, labels ,label in dataloader:
#     # print(images.shape, labels.shape)
#     print(labels)
#     break

# 训练模型
test_model(model, dataloader, criterion, optimizer, device)