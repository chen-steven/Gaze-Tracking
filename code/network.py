import torch.nn as nn
import torch.nn.functional as F
import torch

class GazeEstimationNet(nn.module):
	def __init__(self):
		super(GazeEstimationNet, self).__init__()
		self.conv1 = nn.Conv2D(3,96, kernel_size=11, stride=4)

		self.features1 = nn.Sequential(
			nn.BatchNorm2d(96)
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.ReLU(True)
			nn.Conv2d(96, 256, kernel_size=5, stride=2),
			nn.BatchNorm2d(256)
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.ReLU(True)
			nn.Conv2d(256, 384, kernel_size=3),
			nn.ReLU(True)
		)

		self.conv2 = nn.Conv2d(384, 256, kernel_size=1)

		self.features2 = nn.Sequential(
			nn.Conv2d(384, 384, kernel_size=3),
			nn.ReLU(True)
			nn.Conv2d(384, 256, kernel_size=3),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.ReLU(True),

		)

		self.regression = nn.Sequential(
			#nn.Linear( , 4096), TODO: calculate input features
			nn.ReLU(True),
			nn.Linear(4096, 512),
			nn.ReLU(True),
			nn.Linear(512, 2)
		)


	def forward(self,x):
		x = self.conv1(x)

		#x1 = self.conv
		x = self.features1(x)
		x = self.features2(x)
		x = self.regression(x)
		return x