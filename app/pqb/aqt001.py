import sys
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from statsmodels.tsa import stattools
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA

class Aqt001(object):
    def __init__(self):
        self.name = 'Aqt001'
        # 数据文件格式：编号 日期 星期几 开盘价 最高价 
        # 最低价 收益价 收益
        # Indexcd	Trddt	Daywk	Opnindex	Hiindex	
        # Loindex	Clsindex	Retindex
        self.data_file = 'data/pqb/aqt001_001.txt'
        
    def startup(self):
        print('ARMA模型...')
        #self.simulate_ar2()
        #self.simulate_arima_p_d_q()
        self.arima_demo()
        
    def simulate_ar2(self):
        print('模拟AR(2)')
        alpha1 = 0.666
        alpha2 = -0.333
        wt = np.random.standard_normal(size=1000)
        x = wt
        for t in range(2, len(wt)):
            x[t] = alpha1 * x[t-1] + alpha2 * x[t-2] + wt[t]
        plt.plot(x, c='b')
        plt.show()
        ar2 = stattools.ARMA(x, (2, 0)).fit(disp=False)
        print('p={0} **** {1}; q={2}***{3}; {4} - {5} - {6}'.format(
                    ar2.k_ar, ar2.arparams, ar2.k_ma, ar2.maparams, 
                    ar2.aic, ar2.bic, ar2.hqic)
        )
        arima2_0_0 = ARIMA(x, order=(2, 0, 0)).fit(disp=False)
        print('ARIMA: p={0} **** {1}; q={2}***{3}; {4} - {5} - {6}'. \
                    format(arima2_0_0.k_ar, arima2_0_0.arparams, 
                    arima2_0_0.k_ma, arima2_0_0.maparams, 
                    arima2_0_0.aic, arima2_0_0.bic, 
                    arima2_0_0.hqic)
        )
        resid = arima2_0_0.resid
        # 绘制ACF
        acfs = stattools.acf(resid)
        print(acfs)
        tsaplots.plot_acf(resid, use_vlines=True, lags=30)
        plt.title('ACF figure')
        plt.show()
        pacfs = stattools.pacf(resid)
        print(pacfs)
        tsaplots.plot_pacf(resid, use_vlines=True, lags=30)
        plt.title('PACF figure')
        plt.show()
        
    def simulate_arima_p_d_q(self):
        print('模拟ARIMA(p,d,q)过程')
        np.random.seed(8)
        alpha1 = 1.2
        alpha2 = -0.7
        beta1 = -0.06
        beta2 = -0.02
        w = np.random.standard_normal(size=1000)
        x = w
        for t in range(2, len(w)):
            x[t] = alpha1 * x[t-1] + alpha2*x[t-2] + w[t] + beta1 * w[t-1] + beta2*w[t-2]
        plt.plot(x, c='b')
        plt.title('ARIMA(p, d, q) Figure')
        plt.show()
        # 查看ACF
        acfs = stattools.acf(x)
        print('ARIMA(q,d,q) ACFS:\r\n{0}'.format(acfs))
        tsaplots.plot_acf(x, use_vlines=True, lags=30)
        plt.title('ARIMA(p,d,q) ACF')
        plt.show()
        # ARIMA拟合
        min_ABQIC = sys.float_info.max
        arima_model = None
        break_loop = False
        '''
        for p in range(0, 5):
            if break_loop:
                break
            for q in range(0, 5):
                print('try {0}, d, {1}...'.format(p, q))
                try:
                    arima_p_d_q = ARIMA(x, order=(p, 0, q)).fit(disp=False)
                    print('..... fit ok')
                    if arima_p_d_q.aic < min_ABQIC:
                        print('..... record good model')
                        min_ABQIC = arima_p_d_q.aic
                        arima_model = arima_p_d_q
                        #if 1==p and 1==q:
                        #    break_loop = True
                except Exception as ex:
                    print('.....!!!!!! Exception')
        print('ARIMA: p={0} **** {1}; q={2}***{3}; {4} - {5} - {6}'. \
                    format(arima_model.k_ar, arima_model.arparams, 
                    arima_model.k_ma, arima_model.maparams, 
                    arima_model.aic, arima_model.bic, 
                    arima_model.hqic)
        )
        '''
        arima_model = ARIMA(x, order=(2, 0, 2)).fit(disp=False)
        print('God_View:ARIMA: p={0} **** {1}; q={2}***{3}; {4} - {5} - {6}'. \
                    format(arima_model.k_ar, arima_model.arparams, 
                    arima_model.k_ma, arima_model.maparams, 
                    arima_model.aic, arima_model.bic, 
                    arima_model.hqic)
        )
        resid = arima_model.resid
        # 绘制ACF
        acfs = stattools.acf(resid)
        print(acfs)
        tsaplots.plot_acf(resid, use_vlines=True, lags=30)
        plt.title('ARIMA(p,d,q) ACF figure')
        plt.show()
        pacfs = stattools.pacf(resid)
        print(pacfs)
        tsaplots.plot_pacf(resid, use_vlines=True, lags=30)
        plt.title('ARIMA(p,d,q) PACF figure')
        plt.show()
        
    def arima_demo(self):
        register_matplotlib_converters()
        data = pd.read_csv(self.data_file, sep='\t', index_col='Trddt')
        sh_index = data[data.Indexcd==1]
        sh_index.index = pd.to_datetime(sh_index.index)
        raw_data = sh_index.Clsindex
        train_data = raw_data[:-3]
        close_price = np.log(train_data)
        plt.plot(close_price)
        plt.show()
        print(train_data.head(n=3))
        # ARIMA拟合
        min_ABQIC = sys.float_info.max
        arima_model = None
        for p in range(0, 5):
            for q in range(0, 5):
                print('try {0}, d, {1}...'.format(p, q))
                try:
                    arima_p_d_q = ARIMA(close_price, order=(p, 1, q)).fit(disp=False)
                    print('..... fit ok')
                    if arima_p_d_q.aic < min_ABQIC:
                        print('..... record good model')
                        min_ABQIC = arima_p_d_q.aic
                        arima_model = arima_p_d_q
                except Exception as ex:
                    print('.....!!!!!! {0}'.format(ex))
        print('ARIMA: p={0} **** {1}; q={2}***{3}; {4} - {5} - {6}'. \
                    format(arima_model.k_ar, arima_model.arparams, 
                    arima_model.k_ma, arima_model.maparams, 
                    arima_model.aic, arima_model.bic, 
                    arima_model.hqic)
        )
        resid = arima_model.resid
        # 绘制ACF
        acfs = stattools.acf(resid)
        print(acfs)
        tsaplots.plot_acf(resid, use_vlines=True, lags=30)
        plt.title('ARIMA(p,d,q) ACF figure')
        plt.show()
        pacfs = stattools.pacf(resid)
        print(pacfs)
        tsaplots.plot_pacf(resid, use_vlines=True, lags=30)
        plt.title('ARIMA(p,d,q) PACF figure')
        plt.show()
        # 预测
        y = arima_model.forecast(3)[0] #(len(train_data), len(raw_data), dynamic=True)
        print('预测值：{0}'.format(np.exp(y)))
        print('row_data:{0}'.format(raw_data))
        print('train_data:{0}'.format(train_data))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        