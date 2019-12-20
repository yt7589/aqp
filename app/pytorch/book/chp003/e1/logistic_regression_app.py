#
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets.samples_generator as skds #skds.make_blobs
from pandas import DataFrame
import torch
from torch.autograd import Variable
from logistic_regression_model import LogisticRegressionModel

class LogisticRegressionApp(object):
    def __init__(self):
        self.name = ''

    def run(self):
        print('二分类逻辑回归问题')
        X_train, y_train = self.load_dataset()
        model = LogisticRegressionModel()
        criterion = torch.nn.BCELoss(size_average=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(1000):
            y_hat = model(X_train)
            loss = criterion(y_hat, y_train)
            print('{0}: {1}'.format(epoch, loss.data.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(model.parameters)
        w = model.get_weight().data.numpy()
        print('weight:{0}; {1}'.format(w.shape, w))
        b = model.get_bias().data.numpy()
        print('bias:{0}; {1}'.format(b.shape, b))
        # 学习后曲线
        x0 = np.linspace(-1.0, 0.0, 100)
        y0 = -w[0][0]/w[0][1]*x0 - b[0]/w[0][1]*1.14
        plt.plot(x0, y0, '-b')

        x1 = Variable(torch.tensor([[3.31, 0.21]]))
        y1_hat = model.forward(x1).data[0][0]
        print('y={0}'.format(y1_hat))
        plt.show()

    def load_dataset(self):
        '''
        X, y = skds.make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=[0.8, 1.2])
        np.savetxt('ds_x.csv', X, delimiter=',')
        np.savetxt('ds_y.csv', y, delimiter=',')
        '''
        X = np.loadtxt(open("./ds_x.csv","rb"), delimiter=",", skiprows=0)
        y = np.loadtxt(open("./ds_y.csv","rb"), delimiter=",", skiprows=0)
        # 绘制第一个类别
        idx1 = np.where(y == 0)
        X1 = np.array([X[idx] for idx in idx1[0]])
        plt.scatter(X1[:, 0:1], X1[:, 1:2], c='r')
        # 绘制第二个类别
        idx2 = np.where(y == 1)
        X2 = np.array([X[idx] for idx in idx2[0]])
        plt.scatter(X2[:, 0:1], X2[:, 1:2], c='b', marker='x')
        # 改变数据类型为float32
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y = y.reshape(y.shape[0], 1)
        return Variable(torch.from_numpy(X)), Variable(torch.from_numpy(y))