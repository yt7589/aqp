import sys
import math
import pickle
import app.pqb
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
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import arch.unitroot as unitroot
import arch as arch
from core.statistics.johansen import Johansen
from ann.linear_regression import LinearRegression
from app.pqb.qic_linear_regression import QciLinearRegression

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
        #self.qcilr_demo()
        self.johansen_test_demo()
        #self.sm_johansen_test()

        #qcilr = QciLinearRegression()
        #qcilr.train()
        #data = np.array([[100.0]], dtype=float)
        #rst = qcilr.predict(data)
        #print(rst)

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

    def qcilr_demo(self):
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
        # 以x作为自变量
        w1, x_y_p = self.do_linear_regression(x, y)
        # 将y作为自变量
        w2, y_x_p = self.do_linear_regression(y, x)
        print('xToy={0}({1}) vs yTox={2}({3})'.format(x_y_p, w1[0][0], y_x_p, w2[0][0]))
        if x_y_p < y_x_p:
            print('######## x  为自变量')
            c = w1[0][0] * x - y
        else:
            print('######### y  为自变量')
            c = w2[0][0] * y - x
        plt.title('Final Cointegration Signal')
        plt.plot(c)
        plt.show()

    def do_linear_regression(self, x, y):
        validate_x = validate_y = test_x = test_y = np.array([])
        qcilr = QciLinearRegression(learning_rate=0.01, 
                    epoch=50000, patience=100, 
                    train_x=x, train_y=y, 
                    validate_x=validate_x, validate_y=validate_y, 
                    test_x=test_x, test_y=test_y)
        weights = qcilr.train()
        print('weights:{0}'.format(weights))
        hedge_ratio = weights[0][0]
        c = hedge_ratio * x - y
        # 采用ADF检验 
        resid_adf = unitroot.ADF(c)
        print('stat={0:0.4f} vs 1%_cv={1:0.4f}'.format( \
                    resid_adf.stat, resid_adf.critical_values['1%']))
        if resid_adf.stat < resid_adf.critical_values['1%']:
            print('resid为稳定时间序列 ^_^')
        else:
            print('resid为非稳定时间序列！！！！！')
        return weights, resid_adf.stat

    def johansen_test_demo(self):
        print('johansen test demo')
        start_date = '2016/02/01'
        end_date = '2016/05/30'
        prices_df = pd.read_pickle('./app/pqb/ewa_ewc_df.p')
        prices_df = prices_df.sort_values(by='date').set_index('date')
        x = prices_df.loc[start_date:end_date].values
        print('x.type:{0}; shape:{1}!'.format(type(x), x.shape))
        
        
        jres = coint_johansen(x, det_order=0, k_ar_diff=1)
        print('特征值：{0}'.format(jres.eig))
        print('特征向量：{0}'.format(jres.evec))
        print('Trace statistic:{0}'.format(jres.lr1))
        print('Critical values (90%, 95%, 99%) for trace statistic:{0}'.format(jres.cvt))
        print('Maximum eigenvalue statistic:{0}'.format(jres.lr2))
        print('Critical values (90%, 95%, 99%) for maximum eigenvalue statistic:{0}'.format(jres.cvm))
        print('Order of eigenvalues:{0}'.format(jres.ind))
      
        x1 = x[:,0]
        x2 = x[:,1]
        print('x1 type:{0}'.format(x1))
        x_centered = x - np.mean(x, axis=0)
        johansen = Johansen(x_centered, model=2, significance_level=0)
        eigenvectors, r = johansen.johansen()
        print('r={0}'.format(r))
        print('ev:{0}\r\n{1}\r\n=>{2}'.format(type(eigenvectors), eigenvectors, eigenvectors[:,0]))
        vec = eigenvectors[:, 0]
        vec_min = np.min(np.abs(vec))
        vec = vec / vec_min
        print('vec:{0}; {1}'.format(type(vec), vec))
        start_date_test = start_date
        end_date_test = '2016/08/11'
        plt.title("Cointegrated series")
        portfolio_insample = np.dot(x, vec)
        plt.plot(portfolio_insample, '-')
        #plt.show()
        x_test = prices_df.loc[start_date_test:end_date_test].values
        portfolio_test = np.dot(x_test, vec)
        plt.plot(portfolio_test, '--')
        #plt.show()
        in_sample = np.dot(x, vec)
        mean = np.mean(in_sample)
        std = np.std(in_sample)
        print('mean:{0}; std:{1}'.format(np.mean(in_sample), np.std(in_sample)))
        plt.axhline(y=mean - std, color='r', ls='--', alpha=.5)
        plt.axhline(y=mean, color='r', ls='--', alpha=.5)
        plt.axhline(y=mean + std, color='r', ls='--', alpha=.5)
        plt.savefig('/content/drive/My Drive/fbm/aqp/aqt003_001.png', format='png')
        prices_df.loc[start_date_test:end_date_test].plot(title="Original price series", rot=15)
        plt.savefig('/content/drive/My Drive/fbm/aqp/aqt003_002.png', format='png')
    
    def sm_johansen_test(self):
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
        p = np.zeros((samples,))
        q = np.zeros((samples,))
        r = np.zeros((samples,))
        for t in range(samples):
            p[t] = 0.3*z[t] + w[t]
            q[t] = 0.6*z[t] + w[t]
            r[t] = 0.2*z[t] + w[t]
        endog = np.vstack((p.reshape(samples, 1), q.reshape(samples, 1), r.reshape(samples, 1)))
        print(endog)
        jres = coint_johansen(endog, det_order=0, k_ar_diff=1)
        print('特征值：{0}'.format(jres.eig))
        print('cvt:{0}'.format(jres.cvt))
        


        





    def test001(self):
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
