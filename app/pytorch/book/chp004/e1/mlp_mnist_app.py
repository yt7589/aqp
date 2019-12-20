#
from __future__ import print_function
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.manifold as skmd
from mlp_mnist_model import MlpMnistModel
from npai_ds import NpaiDs
from adam import Adam
from cross_entropy import CrossEntropy
#from ann.loss.negative_log_likelyhood import NegativeLogLikelyhood
from layer import Layer
import layer_common as allc
from dense import Dense
from dropout import Dropout
from activation import Activation
from fclr_layer import FclrLayer
from ces_layer import CesLayer

class MlpMnistApp(object):
    def __init__(self):
        self.name = 'ann.ml.MlpApp'
        
    def run(self):
        print('MLP用于手写数字识别')
        # self.show_mnist_in_tsne()
        # 载入MNIST数据集
        X, y = self.load_mnist_ds()
        n_samples, n_features = X.shape
        n_hidden = 512
        X_raw, X_test, y_raw, y_test = NpaiDs.train_test_split(X, y, test_size=0.2, seed=1)
        X_train, X_validate, y_train, y_validate = NpaiDs.train_test_split(X_raw, y_raw, test_size=0.15, seed=1)
        model = MlpMnistModel(optimizer=Adam(),
                        loss=CrossEntropy,
                        validation_data=(X_validate, y_validate))
        # 添加连接层
        model.add(FclrLayer(optimizer=Adam, K=512, N=784, epsilon=0.1))
        model.add(CesLayer(optimizer=Adam(), K=10, N=512))
        model.summary()
        run_mode = 0
        if 1 == run_mode:
            #
            train_err, val_err = model.fit(X_train, y_train, n_epochs=1, batch_size=256)
            # Training and validation error plot
            n = len(train_err)
            training, = plt.plot(range(n), train_err, label="Training Error")
            validation, = plt.plot(range(n), val_err, label="Validation Error")
            plt.legend(handles=[training, validation])
            plt.title("Error Plot")
            plt.ylabel('Error')
            plt.xlabel('Iterations')
            plt.show()

            _, accuracy = model.test_on_batch(X_test, y_test)
            print ("Accuracy:", accuracy)
            model.save_model()
        else:
            print('进行预测')
            Xt = X_test[0:1, :]
            _, yt = model.predict(Xt)
            yg = y_test[0:1, :]
            print('y_hat:{0}; y:{1}'.format(yt, yg))

        
    def load_mnist_ds(self):
        # CSV文件下载链接：https://www.openml.org/d/554
        # 从网络上获取数据集：X, y = skds.fetch_openml('mnist_784', \
        # version=1, return_X_y=True)
        with open('data/mnist_784.csv', newline='', encoding='UTF-8') as fd:
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
        return X0, y0

    def show_mnist_in_tsne(self):
        X, y_ = self.load_mnist_ds()
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