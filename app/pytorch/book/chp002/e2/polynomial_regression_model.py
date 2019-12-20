import torch
from torch.autograd import Variable

class PolynomialRegressionModel(torch.nn.Module):
    def __init__(self, in_features=1, out_features=1):
        super(PolynomialRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_features, 
                    out_features=out_features, bias=True)

    def get_weights(self):
        return self.linear.weight
    
    def get_biases(self):
        return self.linear.bias

    def forward(self, x):
        return self.linear(x)

    
