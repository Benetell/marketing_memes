import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # fully connected layer
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# set a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# load data
train_dataset = datasets.MNIST(root='mnist_dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True)  # ha végigment minden képen egy epochban akkor shuffle mielőtt másik epoch
test_dataset = datasets.MNIST(root='mnist_dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                         shuffle=True)  # ha végigment minden képen egy epochban akkor shuffle mielőtt másik epoch

# initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    # data = images, targets = correct labels
    for batch_idx, (data, targets) in enumerate(train_loader):
        # data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # get to correct shape
        data = data.reshape(data.shape[0], -1)

        # compute the forward pass to get predictions (scores)
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()  # clear grads from previous minibatch
        loss.backward()

        # gradient descent or adam step
        optimizer.step()  # update the weight


# check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set to evaluation mode (dropout and batch normalization layers off)
    if loader.dataset.train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on testing data")

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device=device)
            labels = labels.to(device=device)
            data = data.reshape(data.shape[0], -1)

            scores = model(data)
            # scores.max() return a two part tuple, by '_' we ignore the first part
            percentage, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

        print(f'got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train() # set back to training mode


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
