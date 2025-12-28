import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

class ArabicDataset(Dataset):
    def __init__(self, img_csv, label_csv, transform=None):
        self.images = pd.read_csv(img_csv, header=None).values
        self.labels = pd.read_csv(label_csv, header=None).values
        self.transform = transform  # حفظ الـ transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        img = img.view(1, 32, 32) / 255.0  # تحويل الصورة لشكل 32x32

        # تحويل الـ Tensor إلى PIL Image قبل تطبيق الـ transform
        to_pil = transforms.ToPILImage()
        img = to_pil(img)  # تحويل الـ Tensor إلى صورة PIL

        # تطبيق الـ transform إذا كان موجود
        if self.transform:
            img = self.transform(img)

        label = int(self.labels[idx]) - 1  # تحويل الـ label
        return img, label
