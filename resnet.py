from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torchmetrics


class ResNetWrapper(pl.LightningModule):
    def __init__(self, num_classes=5):
        super().__init__()
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(num_filters, num_classes)
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

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
        preds = self(x)
        self.train_acc(preds, y)
        self.log('train_acc_step', self.train_acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = self(x)
        self.log('val_loss', loss)
        self.val_acc(logits, y)
        self.log('val_acc_step', self.val_acc, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = logits.argmax(dim=1)
        # Compute any additional metrics here if needed
        return {'test_loss': loss, 'preds': preds, 'labels': y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc)
        self.log('val_acc_epoch', self.val_acc)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
batch_size = 32
num_epochs = 15

if __name__ == '__main__':
    freeze_support()
    # Initialize ResNetWrapper model
    resnet_model = ResNetWrapper(num_classes=100)

    # Move the model to the device
    # resnet_model.to(device)

    # Initialize data module
    data_dir = '/users/bene/codes/'
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


    dm = MyDataModule(train, val, test, batch_size=batch_size, num_workers=2)

    # Initialize ModelCheckpoint callback to save model checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath='/kaggle/working/res',  # Directory to save checkpoints
        filename='model-{epoch:02d}-{val_loss:.2f}',  # Checkpoint filename format
        save_top_k=1,  # Save only the best model based on validation loss
        mode='min',  # Monitoring mode (minimize validation loss)
    )

    # Initialize CSVLogger callback to log metrics to a CSV file
    csv_logger = CSVLogger(save_dir='/kaggle/working/res3_logs', name="res3_log", version = 1)

    # Initialize trainer with callbacks
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=num_epochs,
        precision=16,
        callbacks=[checkpoint_callback],
        val_check_interval=1.0,
        log_every_n_steps=7,
        persistent_workers=True
    )

    # Train the model
    #trainer.fit(resnet_model, dm)
    model = ResNetWrapper.load_from_checkpoint(
        checkpoint_path="/Users/bene/codes/projekt/PyTorch/ResNet/results/model-epoch=14-val_loss",
        hparams_file="/Users/bene/codes/projekt/PyTorch/ResNet/results/hparams.yaml",
        map_location=None,
    )
    # Test the model
    res = trainer.test(model=resnet_model, ckpt_path="/Users/bene/codes/projekt/PyTorch/ResNet/results/model-epoch=14-val_loss"
                                               "=1.79.ckpt",  verbose=True, datamodule=dm)

    print(res[0])
