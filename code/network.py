import torch.nn as nn
import torch.nn.functional as F
import torch

class GazeEstimationNet(nn.module):
	def __init__(self):
		super(GazeEstimationNet, self).__init__()
		self.conv1 = nn.Conv2D(3,96, kernel_size=11, stride=4)

		self.features = nn.Sequential(
			nn.BatchNorm2d(96)
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.ReLU(True)
			nn.Conv2d(96, 256, kernel_size=5, stride=2),

			)
	def forward(self,x):
		pass