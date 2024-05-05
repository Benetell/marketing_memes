import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


class ResNetWrapper(pl.LightningModule):
    def __init__(self, num_classes=5):
        super().__init__()
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Hyperparameters
batch_size = 64
num_epochs = 3

# Initialize ResNetWrapper model
resnet_model = ResNetWrapper(num_classes=5)

# Initialize data module
data_dir = "/users/bened/codes/smaller_dataset/"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train = ImageFolder(data_dir + "train/", transform=transform)
val = ImageFolder(data_dir + "val/", transform=transform)
test = ImageFolder(data_dir + "test/", transform=transform)


# Initialize LightningDataModule
class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size, num_workers):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


dm = MyDataModule(train, val, test, batch_size=batch_size, num_workers=0)

# Initialize ModelCheckpoint callback to save model checkpoints
checkpoint_callback = ModelCheckpoint(
    dirpath='../res1',  # Directory to save checkpoints
    filename='model-{epoch:02d}-{val_loss:.2f}',  # Checkpoint filename format
    save_top_k=1,  # Save only the best model based on validation loss
    mode='min',  # Monitoring mode (minimize validation loss)
)

# Initialize CSVLogger callback to log metrics to a CSV file
csv_logger = CSVLogger(save_dir='../res', name='resnet_logs', version=1)

# Initialize trainer with callbacks
trainer = pl.Trainer(
    min_epochs=1,
    max_epochs=num_epochs,
    precision='bf16-mixed',
    callbacks=[checkpoint_callback],
    logger=csv_logger,
    val_check_interval=1.0,
    log_every_n_steps=7
)

# Train the model
trainer.fit(resnet_model, dm)

# Test the model
trainer.test(resnet_model, dm)
