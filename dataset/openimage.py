import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset

def check_image_path(image_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    _, ext = os.path.splitext(image_path)
    return ext.lower() in image_extensions

class DatasetJson(Dataset):
    def __init__(self, data_path, transform=None, size=256):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.size = size
        # json_path = os.path.join(data_path, 'image_paths.json')
        # assert os.path.exists(json_path), f"please first run: python3 tools/openimage_json.py"
        # with open(json_path, 'r') as f:
        #     self.image_paths = json.load(f)
        print("Start gather image paths")
        self.image_paths = sorted(os.path.join(data_path, i) for i in os.listdir(data_path) if check_image_path(i))
        if not self.image_paths:
            raise ValueError(f"No image files found in {data_path}")
        self.aug = transforms.Resize(size)
        print(f"Found {len(self.image_paths)} images in {data_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')
    
    def getdata(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.aug(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)


def build_openimage(args, transform):
    return DatasetJson(args.data_path, transform=transform, size=args.image_size)
