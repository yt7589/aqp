#
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class DiabetesModel(torch.nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

    def forward(self, x):
        relu = torch.nn.ReLU()
        sigmoid = torch.nn.Sigmoid()
        print('forward.x:{0}'.format(x.shape))
        a1 = relu(self.l1(x))
        a2 = relu(self.l2(a1))
        y_hat = sigmoid(self.l3(a2))
        return y_hat