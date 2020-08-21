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

"""use to evaluate mode on validation data (with labels)"""


def test(model, optimizer, epoch, loss, batch_size, mode):
    # determine whether GPU present
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    top_five = 0
    top_one = 0
    criterion = nn.CrossEntropyLoss()

    # determine whether to load val or train data
    txt_file = ""
    if mode == "val":
        txt_file = "val.txt"
    elif mode == "train":
        txt_file = "train.txt"
    else:
        raise ValueError("not a valid mode")

    # create dataloader with test data
    valid_data = MiniplacesDataset(
        txt_file=txt_file,
        root_dir="/content/images/",
        transform=transforms.Compose(
            [
                transforms.Resize(227),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False
    )

    test_loss = 0.0
    model.eval()

    for batch_idx, (data, target, filename) in enumerate(test_loader):
        # move to GPU
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)

        # get top 5 predictions
        _, pred = torch.topk(output, 5, 1)
        top = pred.cpu().detach().numpy()
        truth = target.data.cpu().detach().numpy()
        for i in range(len(truth)):
            if truth[i] in top[i]:
                top_five += 1
            if truth[i] == top[i][0]:
                top_one += 1
    # average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print("Test Loss: {}".format(round(test_loss, 6)))
    top_5_acc = top_five / len(test_loader.dataset)
    top_1_acc = top_one / len(test_loader.dataset)
    print(
        "Top 5 Accuracy: {}, {}/{}".format(
            top_5_acc, top_five, len(test_loader.dataset)
        )
    )
    print(
        "Top 1 Accuracy: {}, {}/{}".format(top_1_acc, top_one, len(test_loader.dataset))
    )
