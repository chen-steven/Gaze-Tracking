import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import logging
import os
import pandas as pd

class GazeDataset(Dataset):
    def __init__(self, data_csv_path, transforms = None):
        self.path = path
        self.data_csv = pd.read_csv(data_csv_path)
        self.transforms = transforms
        self.data = []

        paths = self.data_csv['path'] + '/spot'+self.data_csv['spot']+'/video'+self.data_csv['video']+self.data_csv['file']



    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        path, target = self.data[idx]
        image = Image.open(path)
        if self.transforms is not None:
            sample = self.transform(sample)
        return sample, target 


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 



def train(train_loader, model, criterion, optimizer, epochs, dev):

    model.train() 

    for i, (images, target) in enumerate(train_loader):
        images = images.to(dev)
        target = target.to(dev)

        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(val_loader, model, criterion, dev):
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(dev)
            target = target.to(dev)

            output = model(images)
            loss = criterion(output, target)


if __name__ == '__main__':
pass





