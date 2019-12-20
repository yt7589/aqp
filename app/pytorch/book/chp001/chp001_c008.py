#
import numpy as np
import matplotlib.pyplot as plt

class Chp001C008(object):
    def __init__(self):
        self.name = 'app.pytorch.book.chp001.Chp001C008'
        self.w = 0.1
        self.learning_rate = 0.1

    def run(self):
        print('numpy求解线性回归')
        X, y = self.load_dataset()
        for epoch in range(100):
            for xi, yi in zip(X, y):
                y_hat = self.forward(xi)
                loss = self.loss(xi, yi)
                grad = self.gradient(xi, yi, y_hat)
                self.w -= self.learning_rate * grad
                print('{0}: loss:{1}; w:{2}'.format(epoch, loss, self.w))
        print('The End ^_^')
        self.draw_dataset(X, y, self.w)

    def forward(self, x):
        return x * self.w
    
    def loss(self, x, y):
        y_hat = self.forward(x)
        return (y_hat - y) * (y_hat - 6)

    def gradient(self, x, y, y_hat):
        return 2.0 * x * (y_hat - y)


    def load_dataset(self):
        delta = np.random.randn(10)*0.06
        X = np.linspace(0, 1, 10)
        y = 2*X + delta
        return X, y

    def draw_dataset(self, X, y, w):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('线性回归数据集')
        plt.scatter(X, y, s=18)
        plt.plot(X, 2*X, '-r')
        plt.plot(X, w*X, '-b')
        plt.show()