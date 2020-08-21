import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from miniplaces_testdataset import MiniplacesTestDataset
from model import Net

'''use to evaluate model on test data with no labels
writes output to a file'''

def evaluate(model, epoch, loss, batch_size):
    # determine whether GPU present
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create dataloader with test data
    test_data = MiniplacesTestDataset(
        root_dir="/content/images/",
        name_of_dir="test",
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
        test_data, batch_size=batch_size, shuffle=False
    )

    model.eval()

    for batch_idx, (data, filename) in enumerate(test_loader):
        # move to GPU
        data = data.to(device)
        output = model(data)

        # get top 5 predictions
        _, pred = torch.topk(output, 5, 1)
        top = pred.cpu().detach().numpy()
        for i in range(len(top)):
            with open("results_test.txt", "a+") as f:
                f.write(
                    filename[i]
                    + " "
                    + str(top[i][0])
                    + " "
                    + str(top[i][1])
                    + " "
                    + str(top[i][2])
                    + " "
                    + str(top[i][3])
                    + " "
                    + str(top[i][4])
                    + "\n"
                )

