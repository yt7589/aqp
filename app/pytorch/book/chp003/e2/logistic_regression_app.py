#
import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets.samples_generator as skds #skds.make_blobs
import pandas as pd
from pandas import DataFrame
import sklearn.manifold as skmd
import torch
from torch.autograd import Variable
from logistic_regression_model import LogisticRegressionModel

class LogisticRegressionApp(object):
    def __init__(self):
        self.name = ''
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def run(self):
        print('softmax回归MNIST应用')
        X_train, y_train = self.load_dataset()
        epochs = 1000
        model = LogisticRegressionModel()
        criterion = torch.nn.BCELoss(size_average=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(epochs):
            y_hat = model(X_train)
            loss = criterion(y_hat, y_train)
            print('{0}: {1}'.format(epoch, loss.data.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        w = model.get_weight().data.numpy()
        print('weight:{0}; {1}'.format(w.shape, w))
        b = model.get_bias().data.numpy()
        print('bias:{0}; {1}'.format(b.shape, b))
        # 测试模型
        idx = 118
        Xt = X_train[idx:idx+1, :]
        yt = y_train[idx:idx+1, :]
        yt_hat = model(Xt)
        print('Xt:{0}; yt:{1}; yt_hat:{2}'.format(Xt.shape, yt.shape, yt_hat.shape))
        r = torch.argmax(yt)
        r_hat = torch.argmax(yt_hat)
        print('r:{0}; r_hat:{1}'.format(r, r_hat))
        plt.title('第{0}张图预测结果: {1}'.format(idx+1, r_hat))
        X_raw = Xt[0].numpy().reshape(28, 28)
        plt.imshow(X_raw, cmap='gray')
        plt.show()

    def load_dataset(self):
        # CSV文件下载链接：https://www.openml.org/d/554
        # 从网络上获取数据集：X, y = skds.fetch_openml('mnist_784', version=1, return_X_y=True)
        with open('E:/alearn/dl/npai/data/mnist_784.csv', newline='', encoding='UTF-8') as fd:
            rows = csv.reader(fd, delimiter=',', quotechar='|')
            X0 = None
            y0 = None
            next(rows)
            cnt = 0
            rst = 0
            amount = 1000 # 每1000条记录保存一次
            X = None
            y = None
            for row in rows:
                x = np.array(row[:784], dtype=np.float)
                x /= 255.0
                y_ = np.array(row[784:])
                if None is X:
                    X = np.array([x])
                    y = np.zeros((1, 10))
                    y[cnt, int(y_[0])] = 1
                else:
                    X = np.append(X, x.reshape(1, 784), axis=0)
                    yi = np.zeros((1, 10))
                    yi[0, int(y_[0])] = 1
                    y = np.append(y, yi.reshape(1, 10), axis=0)
                if cnt % amount == 0 and cnt > 0:
                    if None is X0:
                        X0 = X
                        y0 = y
                    else:
                        X0 = np.append(X0, X, axis=0)
                        y0 = np.append(y0, y, axis=0)
                    X = None
                    y = None
                    cnt = 0
                    rst += amount
                    print('处理完{0}记录'.format(rst))
                else:
                    cnt += 1
            #self.draw_dataset(X0, y0)
            print('X0:{0} vs {2}; y0:{1} vs {3}'.format(X0.shape, y0.shape, X0.dtype, y0.dtype))
            X0 = np.array(X0, dtype=np.float32)
            y0 = np.array(y0, dtype=np.float32)
        return Variable(torch.from_numpy(X0)), Variable(torch.from_numpy(y0))

    def draw_dataset(self, X, y):
        idx = 101
        plt.title('{0}th sample: {1}'.format(idx, np.argmax(y[idx])))
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        self.show_mnist_in_tsne(X, y)
        plt.show()

    def show_mnist_in_tsne(self, X, y_):
        y = np.argmax(y_, axis=1)
        row_embedded = skmd.TSNE(n_components=2).fit_transform(X)
        pos = pd.DataFrame(row_embedded, columns=['X', 'Y'])
        pos['species'] = y
        ax = pos[pos['species']==0].plot(kind='scatter', x='X', y='Y', color='blue', label='0')
        pos[pos['species']==1].plot(kind='scatter', x='X', y='Y', color='red', label='1', ax=ax)
        pos[pos['species']==2].plot(kind='scatter', x='X', y='Y', color='green', label='2', ax=ax)
        pos[pos['species']==3].plot(kind='scatter', x='X', y='Y', color='yellow', label='3', ax=ax)
        pos[pos['species']==4].plot(kind='scatter', x='X', y='Y', color='brown', label='4', ax=ax)
        pos[pos['species']==5].plot(kind='scatter', x='X', y='Y', color='orange', label='5', ax=ax)
        pos[pos['species']==6].plot(kind='scatter', x='X', y='Y', color='black', label='6', ax=ax)
        pos[pos['species']==7].plot(kind='scatter', x='X', y='Y', color='pink', label='7', ax=ax)
        pos[pos['species']==8].plot(kind='scatter', x='X', y='Y', color='purple', label='8', ax=ax)
        pos[pos['species']==9].plot(kind='scatter', x='X', y='Y', color='cyan', label='9', ax=ax)
        plt.show()