from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
def target_to_oh(target):
    NUM_CLASSES = 80  # Number of classes
    one_hot = torch.zeros(NUM_CLASSES, dtype=torch.float32)  # Create a tensor of zeros with shape (NUM_CLASSES,)
    one_hot[target] = 1.0  # Set the correct class index to 1
    return one_hot


class CustomDataset(Dataset):
    def __init__(self, dataset, idx_to_label, transform=None):
        self.dataset = dataset
        self.idx_to_label = idx_to_label
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]) 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if isinstance(img, str):  # If dataset returns path, load the image
            img = Image.open(img).convert('RGB')
        elif isinstance(img, torch.Tensor):
            pass  # Already a tensor, do nothing
        elif isinstance(img, Image.Image):
            img = img  # Already a PIL image
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        if self.transform and not isinstance(img, torch.Tensor):
            img = self.transform(img)

        y_class = target_to_oh(label)
        return img, y_class


import os

class CustomOODDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = sorted(os.listdir(data_path))  # load file names
        self.full_paths = [os.path.join(data_path, f) for f in self.data]

    def __getitem__(self, idx):
        img = Image.open(self.full_paths[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        #print(type(img))
        return img, torch.tensor(1.0)  # 1 = OOD label

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"<CustomOODDataset: {self.data_path} | {len(self)} samples>"
