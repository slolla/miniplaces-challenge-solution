from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MiniplacesDataset(Dataset):
    """Custom Miniplaces dataset."""

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file containing image path and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(txt_file, "r") as f:
            data = f.readlines()
            data = [i.strip().split(" ") for i in data]
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image path from text file and root directory
        img_name = os.path.join(self.root_dir, self.data[idx][0])
        sample = Image.open(img_name)

        # get label from text file
        target = int(self.data[idx][1])

        # transform sample if applicable
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
