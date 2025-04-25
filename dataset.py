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

    # For CustomDataset
    def __getitem__(self, idx):
        try:
            img, label = self.dataset[idx]
            
            # Force conversion to tensor regardless of original type
            if isinstance(img, str):
                # Convert path to image
                img = Image.open(img).convert('RGB')
                img = self.transform(img)
            elif isinstance(img, Image.Image):
                # Convert PIL image to tensor
                img = self.transform(img)
            elif not isinstance(img, torch.Tensor):
                # Any other type - create empty tensor
                print(f"Unexpected type at idx {idx}: {type(img)}")
                img = torch.zeros((3, 224, 224), dtype=torch.float32)
                
            # Double-check it's a tensor before returning
            if not isinstance(img, torch.Tensor):
                print(f"STILL not a tensor after processing: {type(img)}")
                img = torch.zeros((3, 224, 224), dtype=torch.float32)
                
            y_class = target_to_oh(label)
            return img, y_class
        except Exception as e:
            print(f"Error in __getitem__ at idx {idx}: {e}")
            # Return fallback values
            return torch.zeros((3, 224, 224), dtype=torch.float32), torch.zeros(80, dtype=torch.float32)

class CustomOODDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.data = sorted(os.listdir(data_path))
        self.full_paths = [os.path.join(data_path, f) for f in self.data]

    def __getitem__(self, idx):
        # Check if the file is an image
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        file_path = self.full_paths[idx]
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            # Skip non-image files
            return self.__getitem__((idx + 1) % len(self.full_paths))

        # Always return a tensor
        img = Image.open(file_path).convert('RGB')
        img = self.transform(img)  # Convert to tensor
        return img, torch.tensor(1.0)  # 1 = OOD label

    def __len__(self):
        return len(self.data)