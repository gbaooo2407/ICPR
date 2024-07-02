import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os
import glob
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import torchvision.transforms.functional as TF

class AgriculturalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        if train:
            data_path = os.path.join(root, "share", "train")
            disease_folders = [f.name for f in os.scandir(data_path) if f.is_dir()]

            for label, disease in enumerate(disease_folders):
                disease_path = os.path.join(data_path, disease)
                subfolders = [f.path for f in os.scandir(disease_path) if f.is_dir()]

                for subfolder in subfolders:
                    image_files = glob.glob(os.path.join(subfolder, "*.tif"))

                    if len(image_files) == 12:
                        self.image_paths.append(image_files)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_paths = self.image_paths[idx]
        label = self.labels[idx]
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

        if self.transform:
            images = [self.transform(image) for image in images]
            images = torch.stack(images, dim=0)  # Xếp chồng hình ảnh thành tensor

        return images, label