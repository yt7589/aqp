#
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.l1 = torch.nn.Linear(784, 10)

    def get_weight(self):
        return self.l1.weight

    def get_bias(self):
        return self.l1.bias

    def forward(self, x):
        return F.softmax(self.l1(x))