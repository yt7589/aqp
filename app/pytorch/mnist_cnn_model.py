#
import torch
import torch.functional as F
from torch.autograd import Variable
import torch.nn.functional as F

class MnistCnnModel(torch.nn.Module):
    def __init__(self):
        super(MnistCnnModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        conv1 = F.relu(self.mp(self.conv1(x)))
        conv2 = F.relu(self.mp(self.conv2(conv1)))
        conv2_ = conv2.view(in_size, -1)
        fc1 = self.fc1(conv2_)
        return F.log_softmax(fc1)