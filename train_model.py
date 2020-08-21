import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from miniplacesdataset import MiniplacesDataset
from model import Net
import time
from test_model import test

'''train model'''
# gpu usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epochs = 10
model = Net()

# move model to GPU
model = model.to(device)

# define loss, optimizer, and batch_size
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
batch_size = 32
epoch = 0
minimum_validation_loss = np.inf

# UNCOMMENT below code and replace PATH to load existing model checkpoint
"""
PATH = "checkpoint.pt"
checkpoint = torch.load(PATH)
print(checkpoint)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
loss = float(loss.item())
minimum_validation_loss = loss
"""

# create train and validation dataloaders
# REPLACE root_dir if necessary
train_data = MiniplacesDataset(
    txt_file="train.txt",
    root_dir="/content/images/",
    transform=transforms.Compose(
        [
            # resize and normalize images
            transforms.Resize(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
val_data = MiniplacesDataset(
    txt_file="val.txt",
    root_dir="/content/images/",
    transform=transforms.Compose(
        [
            # resize and normalize images
            transforms.Resize(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)
valid_loader = torch.utils.data.DataLoader(
    val_data, batch_size=batch_size, shuffle=True
)

# training loop
for epoch in range(epoch + 1, n_epochs):
    train_loss = 0
    valid_loss = 0
    model.train()
    for batch_index, (data, target, filename) in enumerate(train_loader):
        # move to GPU
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)
        print(epoch, batch_index, "stepped")

    # validation
    model.eval()
    for batch_index, (data, target, filename) in enumerate(valid_loader):
        # move to GPU
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)

    # average loss calculations
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    # Display loss statistics
    print("epoch: {} || training loss: {} || validation loss: {} ".format(epoch, round(train_loss, 6), round(valid_loss, 6)))

    # Save model if loss decreases
    if valid_loss <= minimum_validation_loss:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            # change filename if necessary
            "model.pt",
        )
        minimum_validation_loss = valid_loss
        print("Saving New Model")

        # test model every epoch if loss decreases
        test(model, optimizer, epoch, loss, batch_size, "val")
