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
from ann.linear_regression import LinearRegression

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
        #self.tf2_learn()
        # 我们需要的线性回归模型非常简单，输入层为1维，直接接输出层也是1维，我们需要的是
        # 其接权值
        lr = LinearRegression()
        #lr.train()
        raw_data = pd.DataFrame({'Cylinders':[4], 'Displacement': [140.0], 
            'Horsepower':[86.0], 'Weight': [2790.0], 
            'Acceleration': [15.6], 'Model Year': [82], 'USA': [1.0],
            'Europe':[0.0], 'Japan':[0.0]}
        )
        mu_val = np.load('./work/mu.txt.npy')
        std_val = np.load('./work/std.txt.npy')
        data = lr.norm(raw_data, mu_val, std_val)
        lr.predict(data)

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
