#
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from polynomial_regression_model import PolynomialRegressionModel

class PolynomialRegressionApp(object):
    def __init__(self):
        self.name = ''
        self.low_limit = -0.5
        self.high_limit = 0.5
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('多项式回归目标曲线')

    def run(self):
        rank = 9
        X_train, y_train = self.load_dataset(rank=rank)
        model = PolynomialRegressionModel(in_features=rank, out_features=1)
        criterion = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(8000):
            y_hat = model(X_train)
            loss = criterion(y_hat, y_train)
            if epoch % 50 == 0:
                print('{0}: {1}'.format(epoch, loss.data.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        w = model.get_weights().data
        bias = model.get_biases().data
        print('w:{0}; bias:{1}'.format(w, bias))
        # 绘制学习的曲线
        test_data_x, test_np_x = self.generate_test_data(rank)
        test_input_x = Variable(torch.tensor([test_np_x]))
        test_y_hat = model.forward(test_input_x) 
        y_raw = test_y_hat.detach().numpy()
        y_hat = y_raw.reshape(y_raw.shape[1])
        plt.plot(test_data_x, y_hat, '-r')
        # 绘制数据散点图
        data_x = X_train[:, :1].flatten()
        data_y = self.target_func(data_x)
        plt.scatter(data_x, data_y)
        plt.show()

    def generate_test_data(self, rank):
        raw_data = np.linspace(self.low_limit, 
                    self.high_limit, 100)
        Xt1 = raw_data.reshape(raw_data.shape[0], 1)
        Xt2 = Xt1 * Xt1
        Xt3 = Xt2 * Xt1
        Xt4 = Xt3 * Xt1
        Xt5 = Xt4 * Xt1
        Xt6 = Xt5 * Xt1
        Xt7 = Xt6 * Xt1
        Xt8 = Xt7 * Xt1
        Xt9 = Xt8 * Xt1
        if 1 == rank:
            Xt = Xt1
        elif 2 == rank:
            Xt = np.hstack((Xt1, Xt2))
        else:
            Xt = np.hstack((Xt1, Xt2, Xt3, Xt4, Xt5, 
                        Xt6, Xt7, Xt8, Xt9))
        return raw_data, np.array(Xt, dtype=np.float32)
        
    def load_dataset(self, rank=1):
        np.random.seed(100)
        X1 = np.random.rand(10, 1) * (self.high_limit - \
                    self.low_limit) - (self.high_limit - \
                    self.low_limit) / 2.0
        X2 = X1 * X1
        X3 = X2 * X1
        X4 = X3 * X1
        X5 = X4 * X1
        X6 = X5 * X1
        X7 = X6 * X1
        X8 = X7 * X1
        X9 = X8 * X1
        if 1 == rank:
            X_raw = X1
        elif 2 == rank:
            X_raw = np.hstack((X1, X2)) 
        else:
            X_raw = np.hstack((X1, X2, X3, X4, X5, 
                        X6, X7, X8, X9))    
        X = np.array(X_raw, dtype=np.float32)
        b = 0.0
        y0 = self.target_func(X1)
        y = np.array(y0, dtype=np.float32)
        return Variable(torch.from_numpy(X)), \
                    Variable(torch.from_numpy(y))

    def draw_curve(self):
        x = np.linspace(self.low_limit, self.high_limit, 1000)
        y = self.target_func(x)
        plt.plot(x, y, '-b')
        #plt.show()

    def target_func(self, x):
        return (x + 0.5)*(8.0*x - 1.0)

    
