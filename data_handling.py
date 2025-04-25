from model import *
from transformers import ViTImageProcessor



def target_to_oh(target):
    NUM_CLASSES = 80  # Number of classes
    one_hot = torch.zeros(NUM_CLASSES)  # Create a tensor of zeros with shape (NUM_CLASSES,)
    one_hot[target] = 1  # Set the correct class index to 1
    return one_hot



def create_train_test_split_proportional(dataset, test_ratio=0.1, seed=42, transform=None):
    """
    Create a train-test split proportional to the dataset's labels.

    Args:
        dataset (ImageFolder): The dataset to split.
        test_ratio (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.
        transform: Image transformation for preprocessing.

    Returns:
        train_dataset, test_dataset, idx_to_label: The train/test split datasets and label mapping.
    """
    random.seed(seed)

    # Group samples by label
    label_to_samples = defaultdict(list)
    for sample in dataset.samples:
        label_to_samples[sample[1]].append(sample)

    train_samples = []
    test_samples = []

    # Split the dataset into train and test samples
    for label, samples in label_to_samples.items():
        random.shuffle(samples)
        num_test = int(len(samples) * test_ratio)
        test_samples.extend(samples[:num_test])
        train_samples.extend(samples[num_test:])

    # Create ImageFolder datasets for train and test
    train_dataset = ImageFolder(dataset.root, transform=transform)
    train_dataset.samples = train_samples
    train_dataset.targets = [sample[1] for sample in train_samples]  # Update targets

    test_dataset = ImageFolder(dataset.root, transform=transform)
    test_dataset.samples = test_samples
    test_dataset.targets = [sample[1] for sample in test_samples]  # Update targets

    # Use the class_to_idx from the original dataset for label mapping
    idx_to_label = {v: k for k, v in dataset.class_to_idx.items()}

    return train_dataset, test_dataset, idx_to_label


class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_dataset = None, batch_size=32, num_workers=4, persistent_workers=True):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def train_dataloader(self):
        train_dataset_with_transform = [(x, target_to_oh(y)) for x, y in self.train_data]
  
        train_loader = DataLoader(train_dataset_with_transform, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers = self.persistent_workers)
     
        return train_loader


    def val_dataloader(self):
        val_dataset_with_transform = [(x, target_to_oh(y)) for x, y in self.val_data]
        val_loader = torch.utils.data.DataLoader(
                    val_dataset_with_transform, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers = self.persistent_workers)

        return val_loader        
    def test_dataloader(self):
        test_dataset_with_transform = [(x, target_to_oh(y)) for x, y in self.test_data]
        return DataLoader(test_dataset_with_transform, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers = self.persistent_workers)

transform = transforms.Compose([ 
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

