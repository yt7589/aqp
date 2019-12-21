#
import torch

class AqpBaseOptim(object):
    def __init__(self, parameters, lr=0.1):
        self.name = ''
        self.params = []
        for p in parameters:
            self.params.append(p)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.params:
                p -= p.grad * self.lr

    def zero_grad(self):
        for p in self.params:
                p.grad.zero_()