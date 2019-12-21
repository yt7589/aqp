#
import math
import torch

class MlpMnistModel(torch.nn.Module):
    def __init__(self):
        super(MlpMnistModel, self).__init__()
        #dev = torch.device("cuda") if torch.cuda\
         #           .is_available() else torch\
          #          .device("cpu")
        self.name = ''
        self.weights = torch.nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = torch.nn.Parameter(torch.zeros(10))
        #self.weights.to(dev)
        #self.bias.to(dev)

    def forward(self, xb):
        return xb @ self.weights + self.bias

    def accuracy(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()