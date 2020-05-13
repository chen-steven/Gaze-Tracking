import torch.nn as nn
import torch.nn.functional as F
import torch

class GazeEstimationNet(nn.Module):
    def __init__(self):
        super(GazeEstimationNet, self).__init__()
        
      

        self.features0 = nn.Seqeuntial(
            nn.Conv2d(3,96, kernel_size=11, stride=4)
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(True),
        )

        self.skip_conv1 = nn.Conv2d(96, 256, kernel_size=7, stride=4)

        self.features1 = nn.Sequential(
           
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(256, 384, kernel_size=3),
            nn.ReLU(True)
        )

        self.skip_conv2 = nn.Conv2d(384, 256, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=2)

        self.features2 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(True),

        )

        self.regression = nn.Sequential(
            #nn.Linear(512 *  , 4096), TODO: calculate input features
            nn.ReLU(True),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, 2)
        )


    def forward(self,x):
        x = self.features0(x)
        x1 = self.skip_conv1(x)
        x = self.features1(x)

        # x2 = F.relu(self.skip_conv2(x))

        # x3 = torch.cat((x1,x2),1)
        # x3 = self.skip_conv3(x3)

        # x = self.features2(x)
        # x = torch.cat((x,x3),1)

        # x = torch.flatten(x,1)

        #x = self.regression(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = GazeEstimationNet()
    print(summary(model, (3,227,227)))