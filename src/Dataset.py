import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class PosterDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        
        # 파일 이름에서 IMDB 평점 추출 assuming the format "filename_rating.jpg"
        label = float(self.image_files[idx].split('_')[0])
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)
