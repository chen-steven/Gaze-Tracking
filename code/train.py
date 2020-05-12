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

class GazeDataset(Dataset):
	def __init__(self, path, transforms = None):
		self.path = path
		self.transforms = transforms
		self.data = []
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		pass




