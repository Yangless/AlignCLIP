import os
from torch.utils.data import Dataset
from PIL import Image
import json

class HICODETDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        """
        Args:
            root (str): Path to the dataset root.
            split (str): Dataset split ("train" or "val").
            transforms (callable, optional): Data augmentation and preprocessing transforms.
        """
        self.root = root
        self.split = split
        self.transforms = transforms
        self.data = self.load_annotations()

    def load_annotations(self):
        """
        Load HICO-DET annotations.
        """
        annotations_path = os.path.join(self.root, f"annotations/{self.split}_anno.json")
        images_dir = os.path.join(self.root, f"{self.split}2017")
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        return annotations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = os.path.join(self.root, f"{self.split}2017", entry["file_name"])
        image = Image.open(image_path).convert("RGB")
        target = entry["annotations"]  # Include bounding boxes and labels.
        if self.transforms:
            image = self.transforms(image)
        return image, target
