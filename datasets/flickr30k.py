import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class Flickr30KDataset(Dataset):
    def __init__(self, root, ann_file, split, transform=None):
        """
        Args:
            root (str): Path to the images folder.
            ann_file (str): Path to the annotation file.
            split (str): Dataset split ('train', 'val', or 'test').
            transform (callable, optional): Transformation to apply to the images.
        """
        self.root = root
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with open(ann_file, 'r') as f:
            self.annotations = f.readlines()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        line = self.annotations[idx].strip()
        parts = line.split('\t')
        img_path = os.path.join(self.root, parts[0])
        captions = parts[1:]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, captions

def build(split, args):
    ann_file = os.path.join(args.data_path, f"flickr30k_{split}.txt")
    img_dir = os.path.join(args.data_path, "images")
    return Flickr30KDataset(img_dir, ann_file, split)
