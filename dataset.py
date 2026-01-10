# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# from torchvision import transforms

# class ArabicDataset(Dataset):
#     def __init__(self, img_csv, label_csv, transform=None):
#         self.images = pd.read_csv(img_csv, header=None).values
#         self.labels = pd.read_csv(label_csv, header=None).values
#         self.transform = transform  # حفظ الـ transform

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img = torch.tensor(self.images[idx], dtype=torch.float32)
#         img = img.view(1, 32, 32) / 255.0  # تحويل الصورة لشكل 32x32

#         # تحويل الـ Tensor إلى PIL Image قبل تطبيق الـ transform
#         to_pil = transforms.ToPILImage()
#         img = to_pil(img)  # تحويل الـ Tensor إلى صورة PIL

#         # تطبيق الـ transform إذا كان موجود
#         if self.transform:
#             img = self.transform(img)

#         label = int(self.labels[idx]) - 1  # تحويل الـ label
#         return img, label




import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class ArabicDataset(Dataset):
    def __init__(self, images_csv, labels_csv, transform=None):
        self.images = pd.read_csv(images_csv, header=None).values
        self.labels = pd.read_csv(labels_csv, header=None).values.squeeze()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # الصور عندك 1024 = 32x32
        img = self.images[idx].reshape(32, 32).astype("uint8")

        # نحولو لـ PIL باش transforms يخدمو مزيان
        img = Image.fromarray(img, mode="L")

        if self.transform:
            img = self.transform(img)

        label = int(self.labels[idx]) - 1
        return img, label
