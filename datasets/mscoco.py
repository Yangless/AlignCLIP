import os
from torch.utils.data import Dataset
from PIL import Image

class MSCOCODataset(Dataset):
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
        Load annotations for MSCOCO.
        """
        annotations_path = os.path.join(self.root, f"annotations/captions_{self.split}2017.json")
        images_dir = os.path.join(self.root, f"{self.split}2017")
        # Use pycocotools or another library to parse annotations (simplified here).
        from pycocotools.coco import COCO
        coco = COCO(annotations_path)
        image_ids = coco.getImgIds()
        return [{"image_id": img_id, "file_name": coco.loadImgs(img_id)[0]["file_name"]} for img_id in image_ids]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = os.path.join(self.root, f"{self.split}2017", entry["file_name"])
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, entry["image_id"]
