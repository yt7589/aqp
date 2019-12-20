#
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.l1 = torch.nn.Linear(2, 4)
        self.l2 = torch.nn.Linear(4, 3)
        self.l3 = torch.nn.Linear(3, 1)

    def forward(self, x):
        relu = torch.nn.ReLU()
        sigmoid = torch.nn.Sigmoid()
        a1 = relu(self.l1(x))
        a2 = relu(self.l2(a1))
        y_hat = sigmoid(self.l3(a2))
        return y_hat