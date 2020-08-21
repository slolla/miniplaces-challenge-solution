from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MiniplacesTestDataset(Dataset):
    """Custom Miniplaces Test dataset used to load the test data without labels."""

    def __init__(self, root_dir, name_of_dir, transform=None):
        """
        Args:
            root_dir (string): Directory containing directory with images (e.g. dir/data).
            name of dir (string): name of the directory with images (e.g. test, val, train)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.name_of_dir = name_of_dir
        self.filenames = sorted(os.listdir(os.path.join(root_dir, name_of_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image path from text file and root directory
        filename = os.path.join(self.name_of_dir, self.filenames[idx])
        img_name = os.path.join(self.root_dir, filename)
        sample = Image.open(img_name)

        # transform sample if applicable
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, filename
