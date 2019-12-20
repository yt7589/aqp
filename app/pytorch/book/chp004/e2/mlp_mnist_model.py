#
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class MlpMnistModel(torch.nn.Module):
    def __init__(self):
        super(MlpMnistModel, self).__init__()
        self.l1 = torch.nn.Linear(784, 520)
        self.d1 = torch.nn.Dropout(0.2)
        self.l2 = torch.nn.Linear(520, 320)
        self.l3 = torch.nn.Linear(320, 240)
        self.l4 = torch.nn.Linear(240, 120)
        self.l5 = torch.nn.Linear(120, 10)

    def forward(self, x):
        relu = torch.nn.ReLU()
        # x (batch_index, channel_index, 28, 28) => (batch_index, 784)
        x = x.view(-1, 784)
        z1 = self.l1(x)
        d1 = self.d1(z1)
        a1 = relu(d1)
        #a1 = relu(self.l1(x))
        a2 = relu(self.l2(a1))
        a3 = relu(self.l3(a2))
        a4 = relu(self.l4(a3))
        return self.l5(a4)