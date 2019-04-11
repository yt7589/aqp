import sys
import math
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from statsmodels.tsa import stattools
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA
import arch.unitroot as unitroot
import arch as arch

class Aqt002(object):
    def __init__(self):
        self.name = 'Aqt001'
        # 数据文件格式：编号 日期 星期几 开盘价 最高价 
        # 最低价 收益价 收益
        # Indexcd	Trddt	Daywk	Opnindex	Hiindex	
        # Loindex	Clsindex	Retindex
        # self.data_file = 'data/pqb/aqt002_001.txt'
        self.data_file = '/content/drive/My Drive/aqp/aqt002_001.txt'
        
    def startup(self):
        print('GARCH模型...')
        #self.garch_simulate_demo()
        self.garch_finance_demo()
        
    def garch_simulate_demo(self):
        np.random.seed(1)
        alpha0 = 0.2
        alpha1 = 0.5
        beta1 = 0.3
        samples = 10000 # 样本数量
        w = np.random.standard_normal(size=samples)
        epsilon = np.zeros((samples,), dtype=float)
        sigma = np.zeros((samples,), dtype=float)
        for i in range(2, samples):
            sigma_2 = alpha0 + alpha1 * math.pow(epsilon[i-1], 2) + \
                        beta1 * math.pow(sigma[i-1], 2)
            sigma[i] = math.sqrt(sigma_2)
            epsilon[i] = sigma[i]*w[i]
        plt.title('epsilon signal')
        plt.plot(epsilon)
        plt.show()
        # 绘制epsilonACFS
        acfs = stattools.acf(epsilon)
        tsaplots.plot_acf(epsilon, use_vlines=True, lags=30)
        plt.title('epsilon ACF')
        plt.show()
        # 绘制epsilon pow2 ACF
        acfs2 = stattools.acf(np.power(epsilon, 2))
        tsaplots.plot_acf(np.power(epsilon, 2), use_vlines=True, lags=30)
        plt.title('pow(epsilon,2) ACF')
        plt.show()
        # GARCH拟合
        am = arch.arch_model(epsilon, x=None, mean='Constant', 
                    lags=0, vol='Garch', p=1, o=0, q=1, 
                    power=2.0, dist='Normal', hold_back=None)
        model = am.fit(update_freq=0)
        # GARCH(1,1)参数
        print('############ GARCH(1,1)参数  ###################')
        print('model_type:{0}'.format(type(model)))
        print('mu={0:0.2f}; a0={1:0.2f}; a1={2:0.2f}; b1={3:0.2f}' \
                    .format(model.params['mu'], model.params['omega'], 
                    model.params['alpha[1]'], model.params['beta[1]']   ))
        print('###############################')
        #print(model.summary())
        # 残差信号
        plt.title('GARCH(1,1) resid')
        plt.plot(model.resid)
        plt.show()
        # 残差ACF
        resid_acf = stattools.acf(model.resid)
        tsaplots.plot_acf(model.resid, use_vlines=True, lags=30)
        plt.title('GARCH(1,1) resid ACF')
        plt.show()
        # ADF检验
        resid_adf = unitroot.ADF(model.resid)
        print('stat={0:0.4f} vs 1%_cv={1:0.4f}'.format( \
                    resid_adf.stat, resid_adf.critical_values['1%']))
        if resid_adf.stat < resid_adf.critical_values['1%']:
            print('resid为稳定时间序列 ^_^')
        else:
            print('resid为非稳定时间序列！！！！！')
        # Ljung-Box检验
        resid_ljung_box = stattools.q_stat(stattools.acf( \
                    model.resid)[1:12], len(model.resid))
        resid_lbv = resid_ljung_box[1][-1]
        print('resid_ljung_box_value={0}'.format(resid_lbv))
        # 0.05为显著性水平
        if resid_lbv < 0.05:
            print('resid为平稳时间序列 ^_^')
        else:
            print('resid为非平稳时间序列！！！！！！！')
        # 预测
        y = model.forecast(horizon=3)
        print('########### 预测值  ###############')
        print('len={0}; p1={1:0.3f}; p2={2:0.3f}; p3={3:0.3f}'. \
                    format(len(y.mean.iloc[-1]), y.mean.iloc[-1][0], 
                    y.mean.iloc[-1][1], y.mean.iloc[-1][2]))
        print('##########################')
        
    def garch_finance_demo(self):
        print('拟合上证综指收盘价...')
        register_matplotlib_converters()
        data = pd.read_csv(self.data_file, sep='\t', index_col='Trddt')
        sh_index = data[data.Indexcd==1]
        sh_index.index = pd.to_datetime(sh_index.index)
        sh_return = sh_index.Retindex
        # raw_data = sh_index.Retindex
        raw_data = sh_index.Clsindex
        train_data = raw_data[:-3]
        dcp = np.log(train_data).diff(1)[1:] #close_price # np.power(close_price.diff(1), 2)
        plt.plot(dcp)
        plt.show()
        # GARCH拟合
        am = arch.arch_model(dcp, x=None, mean='Constant', 
                    lags=0, vol='Garch', p=3, o=0, q=2, 
                    power=2.0, dist='Normal', hold_back=None)
        model = am.fit(update_freq=0)
        # GARCH(1,1)参数
        print('############ GARCH(1,1)参数  ###################')
        '''
        print('mu={0:0.2f}; a0={1:0.2f}; a1={2:0.2f}; a2={3:0.2f}, b1={4:0.2f}, b2={5:0.2f}' \
                    .format(model.params['mu'], model.params['omega'], 
                    model.params['alpha[1]'], model.params['alpha[2]'], model.params['beta[1]'], model.params['beta[2]']   ))
        '''
        print('###############################')
        resid = model.resid
        # 绘制ACF
        acfs = stattools.acf(resid)
        print(acfs)
        tsaplots.plot_acf(resid, use_vlines=True, lags=30)
        plt.title('GARCH(p,q) ACF figure')
        plt.show()
        '''
        pacfs = stattools.pacf(resid)
        print(pacfs)
        tsaplots.plot_pacf(resid, use_vlines=True, lags=30)
        plt.title('ARIMA(p,d,q) PACF figure')
        plt.show()
        '''
        # ADF检验
        resid_adf = unitroot.ADF(resid)
        print('stat={0:0.4f} vs 1%_cv={1:0.4f}'.format(resid_adf.stat, resid_adf.critical_values['1%']))
        if resid_adf.stat < resid_adf.critical_values['1%']:
            print('resid为稳定时间序列 ^_^')
        else:
            print('resid为非稳定时间序列！！！！！')
        # Ljung-Box检验
        resid_ljung_box = stattools.q_stat(stattools.acf(resid)[1:12], len(resid))
        resid_lbv = resid_ljung_box[1][-1]
        print('resid_ljung_box_value={0}'.format(resid_lbv))
        # 0.05为显著性水平
        if resid_lbv < 0.05:
            print('resid为平稳时间序列 ^_^')
        else:
            print('resid为非平稳时间序列！！！！！！！')
        # 预测
        frst = model.forecast(horizon=3)
        y = frst.mean.iloc[-1]
        print('预测值：{0}'.format(y))
        p1 = math.exp(math.log(train_data[-1]) + y[0])
        p2 = math.exp(math.log(p1) + y[1])
        p3 = math.exp(math.log(p2) + y[2])
        print('        预测值    实际值  (3957.534)')
        print('第一天：{0} vs 4034.310'.format(p1))
        print('第二天：{0} vs 4121.715'.format(p2))
        print('第三天：{0} vs 4135.565'.format(p3))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        