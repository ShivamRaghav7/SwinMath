import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as f

class HMEDataset(Dataset):
    def __init__(self, csv_path, img_dir, tokenizer, transform=None):
        self.data = pd.read_csv(csv_path).dropna(subset=['label'])
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image']
        latex = str(self.data.iloc[idx]['label'])

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        encoded_text = self.tokenizer.encode(latex)
        target_tensor = torch.tensor(encoded_text, dtype=torch.long)
        
        return image, target_tensor


class ResizeAndPadSquare:
    def __init__(self, target_size=256):
        self.target_size = target_size
    
    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padded_img = f.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=255)
        return f.to_tensor(padded_img)


class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        images = []
        labels = []

        for img, label in batch:
            images.append(img)
            labels.append(label)

        images = torch.stack(images)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.pad_idx
        )
        return images, labels