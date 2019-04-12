import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from statsmodels.tsa import stattools
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA
import arch.unitroot as unitroot
import arch as arch

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

class Aqt003(object):
    def __init__(self):
        self.name = 'Aqt003'

    def startup(self):
        print('交易对协整模型...')
        #self.simulate_demo()
        self.tf2_learn()

    def simulate_demo(self):
        '''
        模拟数据生成
        '''
        # 生成白噪声信号
        samples = 1000
        w = np.random.standard_normal(size=samples)
        # 生成随机游走序列
        z = np.zeros((samples,))
        for t in range(1, samples):
            z[t] = z[t-1] + w[t]
        # 生成非平稳信号，即交易对x和y
        x = np.zeros((samples,))
        y = np.zeros((samples,))
        p = 0.3
        q = 0.6
        for t in range(samples):
            x[t] = p*z[t] + w[t]
            y[t] = q*z[t] + w[t]
        fig = plt.figure(figsize=(6, 6))
        w_plt = plt.subplot(2, 2, 1, title='White Noise: w')
        w_plt.plot(w)
        z_plt = plt.subplot(2, 2, 2, title='Random Walk: z')
        z_plt.plot(z)
        x_plt = plt.subplot(2, 2, 3, title='Non Stationary Signal: x')
        x_plt.plot(x)
        y_plt = plt.subplot(2, 2, 4, title='Non Stationary Signal: y')
        y_plt.plot(y)
        fig.tight_layout()
        plt.show()
        # 生成协整信号
        a = 2
        b = -1
        c = a * x + b * y
        plt.plot(c)
        plt.title('Cointegration Model')
        plt.show()
        # 采用ADF检验 
        resid_adf = unitroot.ADF(c)
        print('stat={0:0.4f} vs 1%_cv={1:0.4f}'.format( \
                    resid_adf.stat, resid_adf.critical_values['1%']))
        if resid_adf.stat < resid_adf.critical_values['1%']:
            print('resid为稳定时间序列 ^_^')
        else:
            print('resid为非稳定时间序列！！！！！')
        # 利用tensorflow linear regression to lean hedge ratio
        print('tensorflow version:{0}'.format(tf.__version__))

    def tf2_learn(self):
        print('学习tensorflow 2.0线性回归程序')
        # 下载数据
        #dataset_path = keras.utils.get_file('auto-mpg.data',
         #                          'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')
        dataset_path = '/Users/arxanfintech/.keras/datasets/auto-mpg.data'
        print(dataset_path)
        column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
        raw_dataset = pd.read_csv(dataset_path, names=column_names,
                         na_values='?', comment='\t',
                         sep=' ', skipinitialspace=True)
        dataset = raw_dataset.copy()
        print(dataset.tail())
        dataset = dataset.dropna()
        origin = dataset.pop('Origin')
        dataset['USA'] = (origin == 1)*1.0
        dataset['Europe'] = (origin == 2)*1.0
        dataset['Japan'] = (origin == 3)*1.0
        print(dataset.tail())
        # 绘制分布图
        '''
        sns.pairplot(dataset[["MPG", "Cylinders", "Displacement"]], diag_kind="kde")
        plt.show()
        sns.pairplot(dataset[["Horsepower", "Weight", "Acceleration"]], diag_kind="kde")
        plt.show()
        '''
        # 求出统计信息
        train_dataset_raw = dataset.sample(frac=0.8,random_state=0)
        train_stats = train_dataset_raw.describe()
        train_stats.pop("MPG")
        train_stats = train_stats.transpose()
        # 准备训练数据集和测试数据集
        train_dataset = self.norm(train_dataset_raw, train_stats['mean'], train_stats['std'])
        train_labels = train_dataset.pop('MPG')
        test_dataset = self.norm(dataset.drop(train_dataset.index), train_stats['mean'], train_stats['std'])
        test_labels = test_dataset.pop('MPG')
        # 创建模型
        model = self.build_model(train_dataset)
        print('summary:{0}'.format(model.summary()))
        # 
        example_batch = train_dataset[:10]
        example_result = model.predict(example_batch)
        print(example_result)
        EPOCHS = 1000
        history = model.fit(
            train_dataset, train_labels,
            epochs=EPOCHS, validation_split = 0.2, verbose=0,
            callbacks=[PrintDot()]
        )
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist.tail())

    def norm(self, x, mu_val, std_val):
        return (x - mu_val) / std_val

    def build_model(self, train_dataset):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mse',
                 optimizer=optimizer,
                 metrics=['mae', 'mse'])
        return model
