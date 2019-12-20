#
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

class Chp001C009(object):
    def __init__(self):
        self.name = ''
        self.w = Variable(torch.tensor([1.0]), requires_grad=True)
        self.learning_rate = 0.01

    def run(self):
        X, y = self.load_dataset()
        for epoch in range(100):
            for xi, yi in zip(X, y):
                y_hat = self.forward(xi)
                loss = self.loss(xi, yi)
                loss.backward()
                #grad = self.gradient(x, y, y_hat)
                self.w.data -= self.learning_rate * self.w.grad.data
                self.w.grad.data.zero_()
                print('{0}: loss:{1}; w:{2}'.format(epoch, loss.data[0], self.w))
        self.draw_dataset(X.numpy(), y.numpy(), self.w.item())
        print('The End ^_^')

    def forward(self, x):
        return x * self.w
    
    def loss(self, x, y):
        y_hat = self.forward(x)
        return (y_hat - y) * (y_hat - y)

    def gradient(self, x, y, y_hat):
        return 2.0 * x * (y_hat - y)

    def load_dataset(self):
        delta = np.random.randn(10)*0.06
        X = np.linspace(0, 1, 10)
        y = 2*X + delta
        return torch.from_numpy(X), torch.from_numpy(y)

    def draw_dataset(self, X, y, w):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('线性回归数据集')
        plt.scatter(X, y, s=18)
        plt.plot(X, 2*X, '-r')
        plt.plot(X, w*X, '-b')
        plt.show()
